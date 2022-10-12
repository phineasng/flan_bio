import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from captum.attr import IntegratedGradients, Saliency, InputXGradient
import numpy as np
from kme.data.utils import REVERSE_TRANSFORM
import os
import numpy as np
from captum.attr import IntegratedGradients, Saliency, InputXGradient
from captum.attr._utils.visualization import visualize_image_attr
from torchvision.transforms import Resize
from sklearn_extra.cluster import KMedoids


def k_pred_FLAN(model, loader, k=10):
    model.eval()
    dataset = loader.dataset

    true_labels = []
    k_predicted_labels = []

    for id in range(len(dataset)):

        img = dataset[id][0].unsqueeze(0)
        true_labels.append(dataset[id][1].tolist()[0])

        features = model._feat_net.process_samples(img)
        feature_norms = torch.norm(features, dim=2, p=2)

        picked_features_id = torch.sort(
            feature_norms, descending=True).indices.squeeze().tolist()[:k]
        picked_features_id.sort()
        picked_features = features[:, picked_features_id, :]

        classifier = model._classifier

        feature_sum = torch.sum(picked_features, 1)
        out = classifier(feature_sum).detach().squeeze()
        k_predicted_labels.append(torch.argmax(torch.softmax(out, 0)).tolist())

    result = pd.DataFrame(
        {'true_labels': true_labels, 'predicted_labels': k_predicted_labels})
    acc = accuracy_score(true_labels, k_predicted_labels)

    return result, acc


def k_pred_pmethods(classifier, features, attr, k):
    pooling = torch.nn.MaxPool2d(kernel_size=4, stride=4)
    # 49 values, 1 for each patch.
    attr_patches = torch.max(pooling(attr), 1).values
    # properly arrange the values (as patch_extractor returns them in feat_net)
    attr_patches = torch.reshape(attr_patches, (-1, 49))  # [1,49]
    # find the k largest values
    picked_id = torch.sort(
        attr_patches.squeeze(), descending=True).indices.tolist()[:k]
    # predict
    picked_id.sort()
    picked_features = features[:, picked_id, :]
    feature_sum = torch.sum(picked_features, 1)
    out = classifier(feature_sum).detach().squeeze()

    return torch.argmax(torch.softmax(out, 0)).tolist()


def comparison_with_acc(model, loader):
    model.eval()
    classifier = model._classifier

    acc_FLAN = []
    acc_intgrad = []
    acc_saliency = []
    acc_xgrad = []

    K = [5, 10, 15, 20, 25, 30, 35, 40, 49]

    for k in K:
        _, acc = k_pred_FLAN(model, loader, k)
        acc_FLAN.append(acc)

        true_labels = []
        ig_predicted_labels = []
        sal_predicted_labels = []
        ixg_predicted_labels = []

        for id in range(len(loader.dataset)):

            img = loader.dataset[id][0].unsqueeze(0)
            true_labels.append(loader.dataset[id][1].tolist()[0])
            target = torch.argmax(model(img), dim=1)
            features = model._feat_net.process_samples(img)

            # Compute IG attributions
            algo = IntegratedGradients(model)
            attr_ig = algo.attribute(img, target=target)
            ig_pred = k_pred_pmethods(classifier, features, attr_ig, k)
            ig_predicted_labels.append(ig_pred)

            # Compute Saliency
            algo = Saliency(model)
            attr_saliency = algo.attribute(img, target=target)
            sal_pred = k_pred_pmethods(classifier, features, attr_saliency, k)
            sal_predicted_labels.append(sal_pred)

            # Compute XGradient
            algo = InputXGradient(model)
            attr_ixg = algo.attribute(img, target=target)
            ixg_pred = k_pred_pmethods(classifier, features, attr_ixg, k)
            ixg_predicted_labels.append(ixg_pred)

        acc_intgrad.append(accuracy_score(true_labels, ig_predicted_labels))
        acc_saliency.append(accuracy_score(true_labels, sal_predicted_labels))
        acc_xgrad.append(accuracy_score(true_labels, ixg_predicted_labels))

    return acc_FLAN, acc_intgrad, acc_saliency, acc_xgrad


def plot_compasrison_with_acc(model, loader):
    acc_FLAN, acc_intgrad, acc_saliency, acc_xgrad = comparison_with_acc(
        model, loader)

    K = [5, 10, 15, 20, 25, 30, 35, 40, 49]

    plt.plot(K, acc_FLAN, label="FLAN")
    plt.plot(K, acc_intgrad, label="int_grad")
    plt.plot(K, acc_saliency, label="saliency")
    plt.plot(K, acc_xgrad, label="Xgrad")
    plt.xticks(K, K)
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for subsets of important features')
    plt.legend()
    plt.show()


def algor_inter(model, loader, id=0):
    model.eval()
    img = loader.dataset[id][0].unsqueeze(0)
    label = torch.argmax(torch.softmax(model(img), dim=1), dim=1).tolist()[0]

    features = model._feat_net.process_samples(img)
    feature_norms = torch.norm(features, dim=2, p=2)
    picked_features_id = torch.sort(
        feature_norms, descending=True).indices.squeeze().tolist()[:5]
    picked_features_id.sort()

    prob = []
    temp = torch.sort(torch.softmax(
        model(img).detach().squeeze(), 0), descending=True)
    temp = {str(int(temp.indices[0])): round(temp.values[0].tolist(), 3), str(int(temp.indices[1])): round(
        temp.values[1].tolist(), 3), str(int(temp.indices[2])): round(temp.values[2].tolist(), 3)}
    top_3_prob = [temp]

    for k in picked_features_id:

        picked_feature = features[:, k, :]
        classifier = model._classifier
        out = classifier(picked_feature).detach().squeeze()

        prob.append(torch.softmax(out, 0)[label].tolist())

        temp = torch.sort(torch.softmax(out, 0), descending=True)
        temp = {str(int(temp.indices[0])): round(temp.values[0].tolist(), 3), str(int(temp.indices[1])): round(
            temp.values[1].tolist(), 3), str(int(temp.indices[2])): round(temp.values[2].tolist(), 3)}
        top_3_prob.append(temp)

    df = pd.DataFrame(top_3_prob, columns=[str(i) for i in range(7)], index=[
        'full model', 'Pixel 1', 'Pixel 2', 'Pixel 3', 'Pixel 4', 'Pixel 5'])
    df.replace(np.nan, '-', regex=True)

    return df


def attribute_image_comparison_skincancer(model, loader, id, device, root):
    folder = os.path.join(root, 'attributions')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, 'comparison_{}.png'.format(id))
    img = loader.dataset[id][0].to(device).unsqueeze(0)
    size = [loader.dataset[id][0].shape[1], loader.dataset[id][0].shape[2]]
    resize = Resize(size)
    unnormalize = REVERSE_TRANSFORM['dermamnist']
    model._feat_net._patch_sz = 4
    target = torch.argmax(model(img), dim=1)

    fold = torch.nn.Fold(
        output_size=28, kernel_size=model._feat_net._patch_sz, stride=model._feat_net._patch_sz)
    latent_repr = model._feat_net.process_samples(img)

    # Compute native attributions
    attr_native = torch.norm(latent_repr, dim=2, p=2).unsqueeze(dim=1)
    attr_native = attr_native.repeat(
        1, 3*(model._feat_net._patch_sz**2), 1)
    attr_native = fold(attr_native)
    attr_native = resize(attr_native).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    # Compute IG attributions
    algo = IntegratedGradients(model)
    attr_ig = algo.attribute(img, target=target)
    attr_ig = resize(attr_ig).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    # Compute Saliency
    algo = Saliency(model)
    attr_saliency = algo.attribute(img, target=target)
    attr_saliency = resize(attr_saliency).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    # Compute XGradient
    algo = InputXGradient(model)
    attr_ixg = algo.attribute(img, target=target)
    attr_ixg = resize(attr_ixg).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    img_numpy = resize(unnormalize(img)).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 4)

    visualize_image_attr(attr_native, img_numpy, use_pyplot=False,
                         method='blended_heat_map', plt_fig_axis=(fig, ax[0]))
    ax[0].set_title('Latent Norms')
    ax[0].title.set_size(8)
    visualize_image_attr(attr_ig, img_numpy, use_pyplot=False,
                         method='blended_heat_map', plt_fig_axis=(fig, ax[1]))
    ax[1].set_title('Integrated Gradients')
    ax[1].title.set_size(8)
    visualize_image_attr(attr_saliency, img_numpy, use_pyplot=False,
                         method='blended_heat_map', plt_fig_axis=(fig, ax[2]))
    ax[2].set_title('Saliency')
    ax[2].title.set_size(8)
    visualize_image_attr(attr_ixg, img_numpy, use_pyplot=False,
                         method='blended_heat_map', plt_fig_axis=(fig, ax[3]))
    ax[3].set_title('InputXGradient')
    ax[3].title.set_size(8)

    #fig.savefig(path, dpi=300, bbox_inches='tight')
    #print('Saved to {}'.format(path))

    plt.show()


def find_neighbors(model, loader, id=0):
    model.eval()
    x_star = loader.dataset[id][0].unsqueeze(0)
    label = torch.argmax(torch.softmax(
        model(x_star), dim=1), dim=1).tolist()[0]

    star_features = model._feat_net.process_samples(x_star).detach()
    sum_star_features = torch.sum(star_features, dim=1)

    # PICK THE NEIGHBORS FROM THE AGGREGATED LATENT SPACE WITH EUKLIDEAN DISTANCE
    sum_sample_features = torch.tensor([])

    for sample in loader.dataset:
        sample_features = model.sample_representation(
            sample[0].unsqueeze(0)).detach()
        sum_sample_features = torch.cat(
            (sum_sample_features, torch.sum(sample_features, dim=1)), dim=0)

    distances = []
    for s in sum_sample_features:
        total_dist = torch.cdist(s.unsqueeze(
            0), sum_star_features.unsqueeze(0), p=2).item()
        distances.append(total_dist)

    furthest_neigh_ind = torch.sort(torch.tensor(
        distances), descending=True).indices[0].tolist()
    nearest_neigh_ind = torch.sort(torch.tensor(
        distances)).indices[1].tolist()

    predictions = torch.tensor([])

    examples = [loader.dataset[id], loader.dataset[nearest_neigh_ind],
                loader.dataset[furthest_neigh_ind]]
    for ex2 in examples:
        predictions = torch.cat(
            (predictions, model(ex2[0].unsqueeze(0)).detach()))

    return examples, predictions


def plot_neighbors(examples, predictions):
    unnormalize = REVERSE_TRANSFORM['dermamnist']
    fig, ax = plt.subplots(1, 3)
    text = ['original', 'nearest neighbor', 'furthest neighbor']

    for i, img in enumerate(examples):
        img_numpy = unnormalize(img[0])
        pred = torch.argmax(torch.softmax(predictions[i], 0)).tolist()
        ax[i].imshow(img_numpy.transpose(0, 1).transpose(1, 2))
        ax[i].set_title(text[i]+' ,pred: ' + str(pred))
        ax[i].title.set_size(6)
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    plt.show()


def global_ex_based(model, loader, no_clusters=3):

    # SEPERATE THE TEST DATA IN no_groups GROUPS:
    for i in range(7):
        locals()['Group' + str(i)] = []
        locals()['Group_real_labels' + str(i)] = []

    medoit_data = {}

    for i, samp in enumerate(loader.dataset):
        img = samp[0].unsqueeze(0)
        out = model(img).detach()
        pred = torch.argmax(torch.softmax(out, dim=1)).tolist()
        group = locals()['Group' + str(pred)]
        group_real_labels = locals()['Group_real_labels' + str(pred)]
        group.append(img)
        group_real_labels.append(samp[1].tolist()[0])

    medoits_model = KMedoids(no_clusters, method='pam')

    def find_class_centers(group, group_labels):

        group_features = torch.tensor([])
        for ex in group:
            x_star_repr = model.sample_representation(ex).detach()
            group_features = torch.cat((group_features, x_star_repr), dim=0)

        group_representation = torch.sum(
            group_features, dim=1)  # aggregated latent space
        df = group_representation.detach().cpu().numpy()
        medoits_model.fit(df)
        medoits_centers = [group[i] for i in medoits_model.medoid_indices_]
        group_labels = [group_labels[i] for i in medoits_model.medoid_indices_]

        return medoits_centers, group_labels

    medoit_data = {}
    for i in range(7):
        medoit_data['Group' +
                    str(i)], medoit_data['Group_real_labels' +
                                         str(i)] = find_class_centers(locals()['Group' + str(i)], locals()['Group_real_labels' + str(i)])

    return medoit_data


def plot_centers(medoit_data,  group_id=0):

    cancer_labels = {'0': 'actinic keratoses and intraepithelial carcinoma',
                     '1': 'basal cell carcinoma', '2': 'benign keratosis-like lesions',
                     '3': 'dermatofibroma', '4': 'melanoma', '5': 'melanocytic nevi',
                     '6': 'vascular lesions'}

    unnormalize = REVERSE_TRANSFORM['dermamnist']

    data = medoit_data['Group' + str(group_id)]
    labels = medoit_data['Group_real_labels' + str(group_id)]
    group_name = cancer_labels[str(group_id)]

    fig, ax = plt.subplots(3, 1)

    for i, img in enumerate(data):
        img_numpy = unnormalize(img[0])
        ax[i].imshow(img_numpy.transpose(0, 1).transpose(1, 2))
        ax[i].set_title('True label: ' + str(cancer_labels[str(labels[i])]))
        ax[i].title.set_size(6)
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    fig.suptitle(group_name, fontsize=10)
    plt.show()
