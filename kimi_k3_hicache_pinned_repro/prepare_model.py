import json
import os
import tempfile
from pathlib import Path


def main():
    source = Path(os.environ["SOURCE_MODEL"]).resolve(strict=True)
    target = Path(os.environ["PRIVATE_MODEL"])
    keep_layers = 5
    resolved_target = target.resolve()
    if target.is_symlink() or (
        resolved_target == source
        or source in resolved_target.parents
        or resolved_target in source.parents
    ):
        raise SystemExit(f"私有模型目录必须与公共模型目录分开：{target}")

    config_name = "config.json"
    config = json.loads((source / config_name).read_text(encoding="utf-8"))
    text_config = config["text_config"]
    original_layers = text_config["num_hidden_layers"]
    if original_layers <= keep_layers:
        raise SystemExit(f"公共模型只有 {original_layers} 个文本层，无法执行五层裁剪")

    attention = text_config["linear_attn_config"]
    for layer_type in ("full_attn_layers", "kda_layers"):
        retained = sorted(layer for layer in attention[layer_type] if 1 <= layer <= keep_layers)
        if not retained:
            raise SystemExit(f"前 {keep_layers} 层未覆盖 {layer_type}，服务未启动")
        print(f"{layer_type}: {retained}", flush=True)

    target.mkdir(parents=True, exist_ok=True)
    private_config = target / config_name
    if private_config.is_symlink() or (private_config.exists() and not private_config.is_file()):
        raise SystemExit(f"私有 config.json 必须是普通文件：{private_config}")

    for entry in source.iterdir():
        if entry.name == config_name:
            continue
        link = target / entry.name
        try:
            link.symlink_to(entry, target_is_directory=entry.is_dir())
        except FileExistsError:
            if not link.is_symlink() or link.resolve() != entry.resolve():
                raise SystemExit(f"私有目录中已有冲突文件：{link}") from None

    text_config["num_hidden_layers"] = keep_layers
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target, prefix=".config-", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump(config, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.close()
            temporary_path.replace(private_config)
        finally:
            temporary_path.unlink(missing_ok=True)
    print(f"私有模型配置：{private_config}，文本层数 {original_layers} → {keep_layers}")


if __name__ == "__main__":
    main()
