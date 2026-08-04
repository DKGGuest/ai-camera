import os
import shutil

def update_model():
    base_dir = "runs/detect"
    if not os.path.exists(base_dir):
        print("No runs found")
        return
        
    models = [d for d in os.listdir(base_dir) if d.startswith('omada_box_model')]
    if not models:
        print("No omada_box_model found")
        return
        
    # Get the latest model directory
    models.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    latest_model_dir = models[0]
    
    best_pt_path = os.path.join(base_dir, latest_model_dir, "weights", "best.pt")
    if os.path.exists(best_pt_path):
        print(f"Copying {best_pt_path} to best.pt")
        shutil.copy(best_pt_path, "best.pt")
        print("Update complete!")
    else:
        print(f"No best.pt found in {latest_model_dir}")

if __name__ == "__main__":
    update_model()
