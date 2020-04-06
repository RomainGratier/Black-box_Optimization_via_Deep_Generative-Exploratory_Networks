from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np 

def mse(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred))

def get_truncated_normal(form, mean=0, sd=1, quant=0.8):
    upp = norm.ppf(quant, mean, sd)
    low = norm.ppf(1 - quant, mean, sd)
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(form)

def generate_sample_from_GAN(target, z, sample_size, generator):
    # Prepare labels
    normalized_target = trainset.scaler.transform(np.array(target).reshape(-1,1))[0]
    z = Variable(FloatTensor(z))
    labels = Variable(FloatTensor(get_truncated_normal(sample_size, mean=normalized_target, sd=1, quant=0.6)))
    images_generated = generator(z, labels)

    return images_generated, labels

def se_between_target_and_prediction(target, x, sample_number, forward, trainset):

    y_labels = np.empty(sample_number)
    y_labels.fill(target)
    y_pred = forward(x.view(-1,28*28)).squeeze(1).cpu().detach().numpy()

    return mse(trainset.scaler.inverse_transform(y_pred.reshape(-1,1)).squeeze(), y_labels), y_pred

def plots_results(target, forward_pred, forward_pred_train, morpho_pred, morpho_pred_train, se, conditions, testset, images_generated, select_img_label_index, fid_value_gen, fid_value_true, nrow=2, ncol=4):

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(8,3), dpi=200)
    i=0
    for row in ax:
        j=0
        if i == 0:
            lst = pd.Series(se)
            n_top_index = lst.nsmallest(ncol).index.values.tolist()
        else:
            imgs = testset.x_data.numpy()[select_img_label_index].squeeze(1)
        for col in row:
            if i == 0:
                image = images_generated[n_top_index[j]]
                col.imshow(image)
                col.axis('off')
                col.set_title(f"Forward={np.round(float(forward_pred[n_top_index[j]]),1)} / true={np.round(float(morpho_pred[n_top_index[j]]),1)} / Cond={np.round(float(conditions[n_top_index[j]]),1)}", fontsize=6)
            else:
                image = imgs[j]
                col.imshow(image)
                col.axis('off')
                col.set_title(f"Forward={np.round(float(forward_pred_train[j]),1)} / true={np.round(float(morpho_pred_train[j]),1)}", fontsize=6)
            j+=1
        i+=1
    plt.suptitle(f'Target : {target} \ FID Value Gen : {np.round(fid_value_gen)} \ FID Value True : {np.round(fid_value_true)}', fontsize=10)
    plt.show()

def compute_fid_mnist_monte_carlo(fake, target, sample_size):
    folder = 'save_data'
    os.makedirs(folder, exist_ok=True)

    # Measure on trained data
    test_img_label = pd.DataFrame(np.around(testset.labels).values.tolist(), columns=['label'])
    random.seed(1)
    select_img_label_index = random.sample(test_img_label[test_img_label['label']==target].index.values.tolist(), sample_size)
    image_from_test = testset.x_data.numpy()[select_img_label_index]

    path_gen = save_numpy_arr(os.path.join(folder, 'gen_img_in_distribution.npy'), fake)
    path_real = save_numpy_arr(os.path.join(folder, 'image_from_test.npy'), image_from_test)

    paths = [path_real, path_gen]
    return calculate_fid_given_paths(paths, mnist_model)

def monte_carlo_inference(target, ncol = 4, nrow =2, sample_number = 100):

    # ------------ Sample z from normal gaussian distribution with a bound ------------
    z = get_truncated_normal((sample_number, latent_dim), quant=0.8)

    # ------------ Generate sample from z and y target ------------
    images_generated, labels = generate_sample_from_GAN(target, z, sample_number, generator)

    # ------------ Compute the mse between the target and the forward model predictions ------------
    se_forward, forward_pred = se_between_target_and_prediction(target, images_generated, sample_number, forward, trainset)

    # Move variable to cpu
    images_generated = images_generated.squeeze(1).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    # ------------ Compare the forward model and the Measure from morphomnist ------------
    with multiprocessing.Pool() as pool:
        thickness = measure_batch(images_generated, pool=pool)['thickness']

    # ------------ Compute the mse between the target and the morpho measure predictions ------------
    se_measure = mse(target, thickness.values)

    # Measure on trained data
    train_img_label = pd.DataFrame(np.around(testset.labels).values.tolist(), columns=['label'])
    select_img_label_index = random.sample(train_img_label[train_img_label['label']==target].index.values.tolist() , ncol)
    image_from_test = testset.x_data.numpy()[select_img_label_index].squeeze(1)

    # ------------ Compute the mse between the target and the forward model predictions ------------
    se_forward_train, forward_pred_train = se_between_target_and_prediction(target, Variable(testset.x_data[select_img_label_index].type(FloatTensor)), ncol, forward, trainset)

    with multiprocessing.Pool() as pool:
        thickness_train = measure_batch(image_from_test, pool=pool)['thickness']

    se_train = mse(target, thickness_train.values)
    # ------------ EDA of the best x* generated ------------
    index = np.argmin(se_forward)
    print()
    print(f" ------------ Best forward image ------------")
    #print(f'Label : {trainset.scaler.inverse_transform(labels[index].reshape(-1, 1)).squeeze()}')
    #print(f"Forward pred = {trainset.scaler.inverse_transform(forward_pred[index].reshape(-1, 1)).squeeze()}")
    #print(f"Morpho measure pred = {thickness.values[index]}")
    print(f"MSE measure pred = {mse(target,thickness.values[index])}")
    print(f"MSE morpho on Generated data: {np.mean(se_measure)}")
    #print(f"MSE forward on Generate data: {np.mean(se_forward)}")
    #print(f"MSE Training data: {np.mean(se_train)}")

    # Transormf output to real value
    model_pred = trainset.scaler.inverse_transform(forward_pred.reshape(-1, 1)).squeeze()
    model_pred_train = trainset.scaler.inverse_transform(forward_pred_train.reshape(-1, 1)).squeeze()
    conditions = trainset.scaler.inverse_transform(labels.reshape(-1, 1)).squeeze()

    # Create true target values
    test_img_label = pd.DataFrame(np.around(testset.labels).values.tolist(), columns=['label'])
    select_img_label_index = random.sample(test_img_label[test_img_label['label']==target].index.values.tolist(), sample_number)
    image_from_test = testset.x_data.numpy()[select_img_label_index]
    
    # Compute FID values
    fid_value_gen = compute_fid_mnist_monte_carlo(np.expand_dims(images_generated, 1), target, sample_number)[0]
    fid_value_true = compute_fid_mnist_monte_carlo(image_from_test, target, sample_number)[0]

    plots_results(target, model_pred, model_pred_train, thickness.values, thickness_train.values, se_forward, conditions, testset, images_generated, select_img_label_index, fid_value_gen, fid_value_true, nrow=2, ncol=4)

    print()
    index = np.argmin(se_measure)
    #print(f" ------------ Best Morpho measured image ------------")
    #print(f'Minimum SE : {np.min(se_measure)}   \\  Label : {trainset.scaler.inverse_transform(labels[index].reshape(-1, 1)).squeeze()}')
    #print(f"Forward pred = {trainset.scaler.inverse_transform(forward_pred[index].reshape(-1, 1)).squeeze()}")
    #print(f"Morpho measure pred = {thickness.values[index]}")
    #print(f"MSE Generated data: {np.mean(se_measure)}")
    #print(f"MSE Training data: {np.mean(mse(target, thickness_train.values))}")

    #plots_results(target, model_pred, model_pred_train, thickness.values, thickness_train.values, se_measure, conditions, testset, images_generated, select_img_label_index, fid_value_gen, fid_value_true, nrow=2, ncol=4)
    
def save_obj_csv(d, path):
    d.to_csv(path+'.csv', index=False)

def load_obj_csv(path):
    return pd.read_csv(path+'.csv')

