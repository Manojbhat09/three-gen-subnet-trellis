import safetensors.torch
from safetensors import safe_open
import torch

def patch_final_layer_adaLN(state_dict, prefix="lora_unet_final_layer", verbose=True):
    """
    Add dummy adaLN weights if missing, using final_layer_linear shapes as reference.
    Args:
        state_dict (dict): keys -> tensors
        prefix (str): base name for final_layer keys
        verbose (bool): print debug info
    Returns:
        dict: patched state_dict
    """
    final_layer_linear_down = None
    final_layer_linear_up = None

    adaLN_down_key = f"{prefix}_adaLN_modulation_1.lora_down.weight"
    adaLN_up_key = f"{prefix}_adaLN_modulation_1.lora_up.weight"
    linear_down_key = f"{prefix}_linear.lora_down.weight"
    linear_up_key = f"{prefix}_linear.lora_up.weight"

    if verbose:
        print(f"\n🔍 Checking for final_layer keys with prefix: '{prefix}'")
        print(f"   Linear down: {linear_down_key}")
        print(f"   Linear up:   {linear_up_key}")

    if linear_down_key in state_dict:
        final_layer_linear_down = state_dict[linear_down_key]
    if linear_up_key in state_dict:
        final_layer_linear_up = state_dict[linear_up_key]

    has_adaLN = adaLN_down_key in state_dict and adaLN_up_key in state_dict
    has_linear = final_layer_linear_down is not None and final_layer_linear_up is not None

    if verbose:
        print(f"   ✅ Has final_layer.linear: {has_linear}")
        print(f"   ✅ Has final_layer.adaLN_modulation_1: {has_adaLN}")

    if has_linear and not has_adaLN:
        dummy_down = torch.zeros_like(final_layer_linear_down)
        dummy_up = torch.zeros_like(final_layer_linear_up)
        state_dict[adaLN_down_key] = dummy_down
        state_dict[adaLN_up_key] = dummy_up

        if verbose:
            print(f"✅ Added dummy adaLN weights:")
            print(f"   {adaLN_down_key} (shape: {dummy_down.shape})")
            print(f"   {adaLN_up_key} (shape: {dummy_up.shape})")
    else:
        if verbose:
            print("✅ No patch needed — adaLN weights already present or no final_layer.linear found.")

    return state_dict


def main():
    print("🔄 Universal final_layer.adaLN LoRA patcher (.safetensors)")
    input_path = input("Enter path to input LoRA .safetensors file: ").strip()
    output_path = input("Enter path to save patched LoRA .safetensors file: ").strip()

    # Load
    state_dict = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    print(f"\n✅ Loaded {len(state_dict)} tensors from: {input_path}")

    # Show all keys that mention 'final_layer' for debug
    final_keys = [k for k in state_dict if "final_layer" in k]
    if final_keys:
        print("\n�� Found these final_layer-related keys:")
        for k in final_keys:
            print(f"   {k}")
    else:
        print("\n⚠️  No keys with 'final_layer' found — will try patch anyway.")

    # Try common prefixes in order
    prefixes = [
        "lora_unet_final_layer",
        "final_layer",
        "base_model.model.final_layer"
    ]
    patched = False

    for prefix in prefixes:
        before = len(state_dict)
        state_dict = patch_final_layer_adaLN(state_dict, prefix=prefix)
        after = len(state_dict)
        if after > before:
            patched = True
            break  # Stop after the first successful patch

    if not patched:
        print("\nℹ️  No patch applied — either adaLN already exists or no final_layer.linear found.")

    # Save
    safetensors.torch.save_file(state_dict, output_path)
    print(f"\n✅ Patched file saved to: {output_path}")
    print(f"   Total tensors now: {len(state_dict)}")

    # Verify
    print("\n🔍 Verifying patched keys:")
    with safe_open(output_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            if "final_layer" in k:
                print(f"   {k}")

        has_adaLN_after = any("adaLN_modulation_1" in k for k in keys)
        print(f"✅ Contains adaLN after patch: {has_adaLN_after}")


if __name__ == "__main__":
    main()
