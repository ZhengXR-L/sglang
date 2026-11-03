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
                             [512 + 128, 1024 + 128, 1536 + 128])
            self.assertEqual(len(plan['eviction_input_ids']), 2816)
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

    def test_l1_hit_and_wrong_answer_fail(self):
        for tiers, ids in [({'device':512,'host':0},[1]),({'device':0,'host':64},[1]),({'device':0,'host':512},[2])]:
            with self.subTest(tiers=tiers,ids=ids),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); a=[7]*640; eviction=[8]*2816
                b.save(root/'plan.json',{'cases':[{'id':'t512-i1','target_host_tokens':512,'input_ids':a,'sha256':b.sha(a),'expected_answer':275}],
                    'eviction_input_ids':eviction,'eviction_sha256':b.sha(eviction)})
                import argparse
                args=argparse.Namespace(output=str(root/'results'),plan=str(root/'plan.json'),base_url='http://localhost',layout='page_first')
                def result(output,details):
                    return {'ttft_ms':100,'first_text_ms':110,'response':{'output_ids':output,'text':'276' if output == [2] else '275','meta_info':{'dp_rank':0,'cached_tokens':0,'cached_tokens_details':details}}}
                with patch.object(b,'http',return_value={}),patch.object(b,'preflight',return_value=65920),patch.object(b.time,'sleep'),patch.object(b,'stream',side_effect=[result([1],{}),result([3],{}),result(ids,tiers)]):
                    with self.assertRaises(ValueError): b.evaluate(args)
                self.assertFalse((root/'results/summary.json').exists())

    def test_token_difference_with_same_answer_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = [7]*640; eviction = [8]*2816
            b.save(root/'plan.json', {'cases':[{'id':'t512-i1','target_host_tokens':512,
                'input_ids':a,'sha256':b.sha(a),'expected_answer':275}],
                'eviction_input_ids':eviction,'eviction_sha256':b.sha(eviction)})
            import argparse
            args = argparse.Namespace(output=str(root/'results'),plan=str(root/'plan.json'),
                                      base_url='http://localhost',layout='page_first')
            def result(ids, tiers):
                return {'ttft_ms':100,'first_text_ms':110,'response':{'output_ids':ids,
                    'text':'275','meta_info':{'dp_rank':0,'cached_tokens':0,
                    'cached_tokens_details':tiers}}}
            with patch.object(b,'http',return_value={}),patch.object(b,'preflight',return_value=3072),\
                 patch.object(b.time,'sleep'),patch.object(b,'stream',side_effect=[
                     result([1],{}),result([3],{}),result([2],{'device':0,'host':512})]):
                b.evaluate(args)
            row = json.loads((root/'results/summary.json').read_text())['rows'][0]
            self.assertFalse(row['token_match'])
            self.assertEqual(row['answer_match'], {'A':True,'R':True})

    def test_resolved_page_layout_required(self):
        info={'hicache_mem_layout':'page_first_kv_split','hicache_io_backend':'kernel_ascend',
              'tp_size':8,'ep_size':8,'speculative_algorithm':None,'hicache_write_policy':'write_through','stream_interval':1,'model_path':'/model',
              'internal_states':[{'memory_usage':{'token_capacity':3072}}]}
        plan={'model_path':'/model','cases':[{'input_ids':[7]*640}]}
        self.assertEqual(b.preflight(info,plan,'page_first'),3072)
        with self.assertRaises(ValueError): b.preflight(info,{'model_path':'/model'},'layer_first')


if __name__=='__main__': unittest.main()
