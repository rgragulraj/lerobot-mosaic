from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id="rgragulraj/mosaic_grasp_square_20260515_161734",
    repo_type="dataset",
    exist_ok=True,
)
api.upload_folder(
    folder_path="/home/rgragulraj/.cache/huggingface/lerobot/rgragulraj/mosaic_grasp_square_20260515_161734",
    repo_id="rgragulraj/mosaic_grasp_square_20260515_161734",
    repo_type="dataset",
    commit_message="upload mosaic grasp square dataset",
)

print("Done")
