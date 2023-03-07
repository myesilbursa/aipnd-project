import unittest
import torch
import model_fnc
import os
import json

data_dir = "flowers"
testing_dir = "testing"
gpu_epochs = 5
cpu_epochs = 3
category_names = "cat_to_name.json"
hidden_units = 400
learning_rate = 0.001
test_image = "flowers/test/37/image_03789.jpg"
correct_prediction_class = "37"
correct_prediction_category = "cape flower"
#num_workers = 4
num_cpu_threads = 16
top_k = 5


def train_test(tester, arch, enable_gpu):

    pin_memory = enable_gpu

    dataloaders, class_to_idx = model_fnc.get_dataloaders(data_dir, enable_gpu, pin_memory)

    model, optimizer, criterion = model_fnc.create_new_model(arch, learning_rate, hidden_units, class_to_idx)
    
    if enable_gpu:
        model.cuda()
    else:
        torch.set_num_threads(num_cpu_threads)

    epochs = gpu_epochs if enable_gpu else cpu_epochs

    model_fnc.train_model(model, criterion, optimizer, epochs,
                    dataloaders['training'], dataloaders['validation'], enable_gpu)

    checkpoint_dir = testing_dir + '/gpu' if enable_gpu else '/cpu'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = checkpoint_dir + '/' + arch + '_checkpoint.pth'

    model_fnc.save_checkpoint(checkpoint, model, optimizer, arch, learning_rate, hidden_units, epochs)


def predict_test(tester, arch, enable_gpu):

    checkpoint_dir = testing_dir + '/gpu' if enable_gpu else '/cpu'
    checkpoint = checkpoint_dir + '/' + arch + '_checkpoint.pth'

    model = model_fnc.load_checkpoint(checkpoint)

    if enable_gpu:
        model.cuda()

    probs, classes = model_fnc.predict(test_image, model, enable_gpu, top_k)

    tester.assertEqual(len(classes), top_k, 'Incorrect number of results')
    tester.assertEqual(classes[0], correct_prediction_class, 'Incorrect prediction')

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    tester.assertEqual(cat_to_name[classes[0]], correct_prediction_category, 'Incorrect prediction')


class TrainingGpuTestCase(unittest.TestCase):
    def test_vgg13(self):
        train_test(self, "vgg13", True)


class TrainingCpuTestCase(unittest.TestCase):
    def test_vgg13(self):
        train_test(self, "vgg13", False)


class InferenceGpuTestCase(unittest.TestCase):
    def test_vgg13(self):
        predict_test(self, "vgg13", True)


class InferenceCpuTestCase(unittest.TestCase):
    def test_vgg13(self):
        predict_test(self, "vgg13", False)


if __name__ == '__main__':
    unittest.main()
