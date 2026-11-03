import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('bench.py'))
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


class Response:
    def __init__(self, events): self.events=events
    def __enter__(self): return iter(self.events)
    def __exit__(self,*args): pass


class Tests(unittest.TestCase):
    def test_prepare_builds_plan_without_external_test_results(self):
        class Tokenizer:
            bos_token_id = 1
            all_special_ids = [0, 1]
            vocab_size = 256

            def encode(self, text, add_special_tokens=False):
                return [2 + ord(char) % 254 for char in text]

        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'transformers': transformers}):
            plan_path = Path(tmp) / 'plan.json'
            b.prepare(tmp, plan_path, repeats=1)
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            self.assertEqual(plan['generator'], 'builtin_arithmetic_v1')
            self.assertEqual([len(case['input_ids']) for case in plan['cases']],
                             [8192 + 128, 32768 + 128, 61440 + 128])
            self.assertEqual(len(plan['eviction_input_ids']), 65536)
            self.assertTrue(all(b.sha(case['input_ids']) == case['sha256'] for case in plan['cases']))

    def test_metadata_is_not_first_token(self):
        events=[b'data: {"meta_info": {}}\n',
                b'data: {"output_ids": [1], "text": "x", "meta_info": {}}\n',b'data: [DONE]\n']
        with patch.object(b.urllib.request,'urlopen',return_value=Response(events)),patch.object(b.time,'perf_counter',side_effect=[0,.1,.2,.3]):
            result=b.stream('http://localhost',{})
        self.assertAlmostEqual(result['ttft_ms'],100)

    def test_empty_zero_and_missing_tokens_fail(self):
        for obj in ({'text':'x'}, {'output_ids':[0],'text':'x'}, {'output_ids':[1],'text':''}):
            events=[('data: '+json.dumps(obj)+'\n').encode(),b'data: [DONE]\n']
            with self.subTest(obj=obj),patch.object(b.urllib.request,'urlopen',return_value=Response(events)):
                with self.assertRaises(ValueError): b.stream('http://localhost',{})

    def test_l1_hit_and_output_corruption_fail(self):
        for tiers, ids in [({'device':8192,'host':0},[1]),({'device':0,'host':6144},[1]),({'device':0,'host':8192},[2])]:
            with self.subTest(tiers=tiers,ids=ids),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); a=[7]*8320; eviction=[8]*65536
                b.save(root/'plan.json',{'cases':[{'id':'t8192-i1','target_host_tokens':8192,'input_ids':a,'sha256':b.sha(a)}],
                    'eviction_input_ids':eviction,'eviction_sha256':b.sha(eviction)})
                import argparse
                args=argparse.Namespace(output=str(root/'results'),plan=str(root/'plan.json'),base_url='http://localhost',layout='page_first')
                def result(output,details):
                    return {'ttft_ms':100,'first_text_ms':110,'response':{'output_ids':output,'meta_info':{'dp_rank':0,'cached_tokens':0,'cached_tokens_details':details}}}
                with patch.object(b,'http',return_value={}),patch.object(b,'preflight',return_value=65920),patch.object(b.time,'sleep'),patch.object(b,'stream',side_effect=[result([1],{}),result([3],{}),result(ids,tiers)]):
                    with self.assertRaises(ValueError): b.evaluate(args)
                self.assertFalse((root/'results/summary.json').exists())

    def test_resolved_page_layout_required(self):
        info={'hicache_mem_layout':'page_first_kv_split','hicache_io_backend':'kernel_ascend','enable_dp_attention':True,'enable_dp_lm_head':True,
              'tp_size':8,'attn_dp_size':8,'speculative_algorithm':None,'hicache_write_policy':'write_through','stream_interval':1,'model_path':'/model',
              'internal_states':[{'memory_usage':{'token_capacity':65920}} for _ in range(8)]}
        self.assertEqual(b.preflight(info,{'model_path':'/model'},'page_first'),65920)
        with self.assertRaises(ValueError): b.preflight(info,{'model_path':'/model'},'layer_first')


if __name__=='__main__': unittest.main()
