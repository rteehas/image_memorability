import unittest
import os
import shutil
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms # Required by data_transforms

# Assuming train_memorability_model.py is in the same directory or accessible in PYTHONPATH
from train_memorability_model import LaMemDataset, get_memorability_model, load_model, data_transforms

class TestTrainingScriptComponents(unittest.TestCase):

    def setUp(self):
        """Set up temporary data for tests."""
        self.test_img_dir = 'temp_test_images'
        self.test_csv_file = 'temp_test_annotations.csv'
        self.test_model_path = 'temp_test_model.pth'
        self.num_dummy_images = 3

        # Create dummy image directory
        if not os.path.exists(self.test_img_dir):
            os.makedirs(self.test_img_dir)

        # Create dummy images and annotations
        self.dummy_annotations = []
        for i in range(self.num_dummy_images):
            img_name = f'test_image_{i}.png'
            img_path = os.path.join(self.test_img_dir, img_name)
            try:
                # Create a small, simple PNG image
                img = Image.new('RGB', (60, 30), color = 'red')
                img.save(img_path)
                self.dummy_annotations.append({'filename': img_name, 'memorability': 0.5 + (i * 0.1)})
            except Exception as e:
                # If image creation fails (e.g. PIL not fully available in a restricted env)
                # This test suite might not be runnable, but we try.
                print(f"Warning: Could not create dummy image {img_path}: {e}")


        # Create dummy CSV file
        if self.dummy_annotations: # Only create CSV if images were likely created
             df = pd.DataFrame(self.dummy_annotations)
             df.to_csv(self.test_csv_file, index=False)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def tearDown(self):
        """Clean up temporary data after tests."""
        if os.path.exists(self.test_img_dir):
            shutil.rmtree(self.test_img_dir)
        if os.path.exists(self.test_csv_file):
            os.remove(self.test_csv_file)
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)

    def test_lamem_dataset(self):
        """Test LaMemDataset initialization, length, and item retrieval."""
        if not os.path.exists(self.test_csv_file) or not os.path.isdir(self.test_img_dir):
            self.skipTest("Skipping dataset test as dummy data was not created (possibly due to PIL issue).")

        dataset = LaMemDataset(csv_file=self.test_csv_file, 
                               img_dir=self.test_img_dir, 
                               transform=data_transforms['val']) # Use 'val' transform for predictable size
        
        self.assertEqual(len(dataset), self.num_dummy_images)
        
        if len(dataset) > 0:
            img_tensor, score = dataset[0]
            self.assertIsInstance(img_tensor, torch.Tensor)
            # Shape after Resize(256,256) and CenterCrop(224)
            self.assertEqual(img_tensor.shape, (3, 224, 224)) 
            self.assertIsInstance(score, torch.Tensor)
            # Check if score is a scalar tensor
            self.assertTrue(score.ndim == 0 or score.shape == torch.Size([])) 
            # Check if the score matches the first dummy annotation if needed (requires more careful setup)
            # For now, just checking type and shape is fine.
            self.assertAlmostEqual(score.item(), self.dummy_annotations[0]['memorability'], places=5)


    def test_get_memorability_model(self):
        """Test model creation and properties."""
        model = get_memorability_model(self.device)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(model.fc.out_features, 1)

        # Check frozen layers (all params before model.fc)
        fc_params_ids = {id(p) for p in model.fc.parameters()}
        
        for name, param in model.named_parameters():
            if id(param) not in fc_params_ids:
                self.assertFalse(param.requires_grad, f"Parameter {name} should be frozen but is not.")
            else: # fc layer parameters
                self.assertTrue(param.requires_grad, f"Parameter {name} in fc layer should not be frozen but is.")


    def test_save_and_load_model(self):
        """Test saving a model state and loading it back."""
        model_to_save = get_memorability_model(self.device)
        torch.save(model_to_save.state_dict(), self.test_model_path)
        self.assertTrue(os.path.exists(self.test_model_path))

        loaded_model = load_model(self.test_model_path, self.device)
        self.assertIsInstance(loaded_model, torch.nn.Module)
        
        # Ensure model is in eval mode after loading
        self.assertFalse(loaded_model.training, "Loaded model should be in evaluation mode.")

        # Perform a simple inference test
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = loaded_model(dummy_input)
        self.assertEqual(output.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
