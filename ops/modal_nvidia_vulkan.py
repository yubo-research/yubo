def nvidia_vulkan_icd_script() -> str:
    return """\
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
elif [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
else
  unset VK_ICD_FILENAMES
fi
export VK_DRIVER_FILES="${VK_ICD_FILENAMES:-}"
export __VK_LAYER_NV_optimus=NVIDIA_only
export __NV_PRIME_RENDER_OFFLOAD=1
"""
