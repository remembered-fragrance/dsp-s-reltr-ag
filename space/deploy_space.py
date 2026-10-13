"""Đẩy thư mục Space đã đóng gói lên Hugging Face (GitHub Actions gọi; không in token).

Biến môi trường: HF_TOKEN (secret GitHub, quyền ghi Space), HF_SPACE_ID (vd org/dsp-s-demo),
DSP_MODEL_REPO (tùy chọn: đặt thành biến của Space để app biết tải trọng số từ đâu).
  python space/deploy_space.py build/space
"""
import os
import sys

from huggingface_hub import HfApi


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "build/space"
    token, space = os.environ.get("HF_TOKEN"), os.environ.get("HF_SPACE_ID")
    if not token or not space:
        sys.exit("cần HF_TOKEN và HF_SPACE_ID")
    api = HfApi(token=token)
    # luôn private; exist_ok để lần sau chỉ cập nhật
    api.create_repo(space, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    if not api.space_info(space).private:
        sys.exit(f"{space} đang PUBLIC: dừng deploy, chuyển Space sang private trước")
    model_repo = os.environ.get("DSP_MODEL_REPO")
    if model_repo:
        api.add_space_variable(space, "DSP_MODEL_REPO", model_repo)
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    info = api.upload_folder(folder_path=folder, repo_id=space, repo_type="space", delete_patterns="*",
                             commit_message=f"deploy từ GitHub {sha}")
    print(f"đã deploy {space} ({info.oid[:7] if hasattr(info, 'oid') else ''})")


if __name__ == "__main__":
    main()
