import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import LeNet,Inception
from PIL import Image


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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    classes = ['cat', 'dog']

    image = Image.open('cat_prove1.jpg')

    normalize = transforms.Normalize(mean=[0.00017, 0.00015, 0.00016], std=[0.10044, 0.08870, 0.07845])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_transform(image)

    #添加批次维度
    #原始张量形状：假设输入的 image 是单张图像，其形状可能是 [C, H, W]（通道数、高度、宽度），例如 ``（RGB图像）。
    #操作后形状：使用 unsqueeze(0) 后，形状变为 [1, C, H, W]，即新增了一个批次维度（batch dimension）。此时数据表示包含1个样本的批次。
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pred = torch.argmax(output, dim=1)
        #item()将单元素张量转换为Python标量,并会将转化后数据移动到CPU中
        result=pred.item()
    #print(pred)
    print("predict value: ", classes[result])


