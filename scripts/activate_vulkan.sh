#!/usr/bin/env bash
# Source this file from the repository root before using ManiSkill RGB rendering.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$PWD/configs/nvidia_icd_egl.json}"
