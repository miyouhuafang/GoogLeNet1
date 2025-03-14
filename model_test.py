import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import LeNet,Inception


def test_data_process():
    root_train_path = r'D:\pycharm\GoodLeNet\cat and dog\data\cats_and_dogs\test'

    normalize = transforms.Normalize(mean=[0.161, 0.149, 0.137], std=[0.057, 0.051, 0.047])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # 加载dataset
    test_data = ImageFolder(root=root_train_path, transform=test_transform)


    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=0
    )
    return test_loader

def test_model_process(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_accuracy = 0
    test_num = 0

    #只进行前向传播计算，不计算梯度，加快运行速度，节省内存
    with torch.no_grad():
        for test_data_x,test_data_y in test_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            pred = torch.argmax(output, dim=1)
            test_accuracy += torch.sum(pred == test_data_y.data)
            test_num += test_data_x.size(0)

            test_acc = test_accuracy.double().item() / test_num
            print(f'Test Accuracy: {test_acc}')


if __name__ == '__main__':
    model = LeNet(Inception)
    model.load_state_dict(torch.load('best_model.pth'))

    #利用现有模型进行模型的测试
    test_loader = test_data_process()
    #test_model_process(model, test_loader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    classes = ['cat', 'dog']
    with torch.no_grad():
        for b_x, b_y in test_loader:
           b_x = b_x.to(device)
           b_y = b_y.to(device)
           #验证模型
           model.eval()
           output = model(b_x)
           pred = torch.argmax(output, dim=1)
           result=pred.item()
           label=b_y.item()
           print("predict value: ",classes[result],'---',"True value ",classes[label])

