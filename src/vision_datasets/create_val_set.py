import torch
import utils
from utils import set_seed
from finetune import creating_new_finetune_exp
from modeling import ImageEncoder, ImageClassifier
from heads import get_classification_head
from vision_datasets.registry import get_dataset
from vision_datasets.common import maybe_dictionarize



if __name__ == '__main__':
    set_seed(seed=42)

    exp_name = 'Try'
    data_location = '../data'
    #models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    model_type = 'ViT-B-16'
    datasets_to_eval = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
    datasets_to_eval = ['SVHN']
    datasets_to_train = ['SVHN']

    batch_size = 128
    lr = 1e-5

    args = creating_new_finetune_exp(data_location=data_location,
                                     model_type=model_type,
                                     exp_name=exp_name,
                                     datasets=datasets_to_eval,
                                     batch_size=batch_size,
                                     lr=lr)

    # Fetching the Image Encoder
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    # Evaluate zero-shot model
    classification_head = get_classification_head(args, dataset=datasets_to_eval[0])
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()
    dataset = get_dataset(
        datasets_to_eval[0],
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )

    device = args.device

    with torch.no_grad():
        for dataloader, data_type in [(dataset.val_loader, 'val'), (dataset.train_loader, 'train'), (dataset.test_loader, 'test')]:
            top1, correct, n = 0., 0., 0.
            for i, data in enumerate(dataloader):
                data = maybe_dictionarize(data)
                x = data['images'].to(device)
                y = data['labels'].to(device)
                logits = utils.get_logits(x, model)
                pred = logits.argmax(dim=1, keepdim=True).to(device)
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            top1 = 100 * correct / n

            print(f'Done evaluating on {datasets_to_eval[0]} {data_type}. Accuracy: {top1:.2f}%')

