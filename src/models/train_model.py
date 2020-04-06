def save_numpy_arr(path, arr):
    np.save(path, arr)
    return path

def generate_sample(minimum, maximum, sample_size):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (sample_size, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.random.uniform(minimum, maximum, sample_size)
    labels = Variable(FloatTensor(labels))
    return generator(z, labels)

def compute_fid_for_mnist(gen_imgs, n_row, img_size, in_distribution_index, out_distribution_index, sample_size):
    gen_img_in_distribution = generate_sample(0, dataset.maximum, sample_size)
    gen_img_out_distribution = generate_sample(dataset.maximum, 1, sample_size)

    random_id_in_distribution = random.sample(index_in_distribution.tolist(), sample_size)
    random_id_out_distribution = random.sample(index_out_distribution.tolist(), sample_size)
    real_imgs_in_distribution = real_dataset[random_id_in_distribution].numpy()
    real_imgs_out_distribution = real_dataset[random_id_out_distribution].numpy()

    folder = 'save_data'
    os.makedirs(folder, exist_ok=True)

    path_gen_in = save_numpy_arr(os.path.join(folder, 'gen_img_in_distribution.npy'), gen_img_in_distribution.cpu().detach().numpy())
    path_gen_out = save_numpy_arr(os.path.join(folder, 'gen_img_out_distribution.npy'), gen_img_out_distribution.cpu().detach().numpy())
    path_real_in = save_numpy_arr(os.path.join(folder, 'real_imgs_in_distribution.npy'), real_imgs_in_distribution)
    path_real_out = save_numpy_arr(os.path.join(folder, 'real_imgs_out_distribution.npy'), real_imgs_out_distribution)

    paths = [path_real_in, path_gen_in]
    fid_value_in_distribution = calculate_fid_given_paths(paths, mnist_model)
    paths = [path_real_out, path_gen_out]
    fid_value_out_distribution = calculate_fid_given_paths(paths, mnist_model)

    return fid_value_in_distribution, fid_value_out_distribution

def sample_image(n_row, batches_done, in_distribution_index, out_distribution_index):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    ## -------------- In distribution --------------
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in np.arange(0, 1, 1/n_row)])
    labels = Variable(FloatTensor(labels))#Variable(LongTensor(labels))

    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

    with multiprocessing.Pool() as pool:
        measure = measure_batch(gen_imgs.squeeze(1).cpu().detach().numpy(), pool=pool)
        thickness = measure['thickness'].values.reshape((n_row, n_row)).mean(axis=0)
    
    label_target = dataset.scaler.inverse_transform(np.array([num for num in np.arange(0, 1, 1/n_row)]).reshape(-1,1)).squeeze()
    mse_generator = mse(label_target, thickness)

    fid_value_in_distribution, fid_value_out_distribution  = compute_fid_for_mnist(gen_imgs, n_row, img_size, in_distribution_index, out_distribution_index, 500)

    print()
    print(f"The thickness distribution =\n{dataset.scaler.transform(thickness.reshape(-1,1)).squeeze()}")
    print(f"Average MSE In dist = {np.mean(mse_generator[in_distribution_index])} \ Average MSE Out dist = {np.mean(mse_generator[out_distribution_index])}")
    print(f"FID score in distribution : mean = {np.around(fid_value_in_distribution[0], decimals=4)} \ std = {np.around(fid_value_in_distribution[1], decimals=4)}")
    print(f"FID score out distribution : mean = {np.around(fid_value_out_distribution[0], decimals=4)} \ std = {np.around(fid_value_out_distribution[1], decimals=4)}")

    return mse_generator, fid_value_in_distribution[0], fid_value_out_distribution[0]

def save_model_check(dist, df_check, mean_out, best_res, df_acc_gen, path_generator):
    if df_check is not None:
        if mean_out < df_check[f'mse_{dist}'].iloc[-1]:
            print(f" ---------- Better Results {dist} distribution of : {df_check[f'mse_{dist}'].iloc[-1] - mean_out} ---------- ")
            torch.save(generator, os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))
            save_obj_csv(df_acc_gen, os.path.join(path_generator, f"results_{dist}_distribution"))

            best_res = mean_out
            df_check = None

    else:
        if mean_out < best_res:
            print(f" ---------- Model Improving {dist} distribution of : {best_res - mean_out}---------- ")
            torch.save(generator, os.path.join(path_generator, f"best_generator_{dist}_distribution.pth"))
            save_obj_csv(df_acc_gen, os.path.join(path_generator, f"results_{dist}_distribution"))

            best_res = mean_out

    return df_check, best_res

def check_memory_cuda():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    #print(f'total    : {info.total}')
    print(f' --------------- MEMORY free     : {info.free} --------------- ')
    #print(f'used     : {info.used}')

def train_loop():
    mse_gan_in_distribution = []
    mse_gan_out_distribution = []
    df_acc_gen = pd.DataFrame(columns=['mse_in', 'mse_out', 'fid_in', 'fid_out'])

    if os.path.exists(path_generator):
        df_check_in_distribution = load_obj_csv(os.path.join(path_generator, 'results_in_distribution'))
        df_check_out_distribution = load_obj_csv(os.path.join(path_generator, 'results_out_distribution'))
    else:
        os.makedirs(path_generator)
        df_check_in_distribution = None
        df_check_out_distribution = None

    best_res_in = 100000
    best_res_out = 100000

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            ## Initialization
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(FloatTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(FloatTensor(np.random.rand(batch_size))) 

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:

                print(
                  "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                  % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                # Delete useless data from GPU
                check_memory_cuda()
                del valid; del fake; del real_imgs; del labels; del z; del gen_labels; del g_loss; del d_loss; del gen_imgs; del validity;
                torch.cuda.empty_cache()
                check_memory_cuda()

                mse_gan, fid_in, fid_out = sample_image(n_row, batches_done, in_distribution_index, out_distribution_index)

                mean_in_mse = np.mean(mse_gan[in_distribution_index])
                mean_out_mse = np.mean(mse_gan[out_distribution_index])

                mse_gan_in_distribution.append(mse_gan[in_distribution_index])
                mse_gan_out_distribution.append(mse_gan[out_distribution_index])

                df = pd.DataFrame([mean_in_mse], columns=['mse_in'])
                df['mse_out'] = mean_out_mse
                df['fid_in'] = fid_in
                df['fid_out'] = fid_out

                df_acc_gen = df_acc_gen.append(df, ignore_index=True)

                # Check if we have better results
                df_check_in_distribution, best_res_in = save_model_check('in', df_check_in_distribution, df['mse_in'].values, best_res_in, df_acc_gen, path_generator)
                df_check_out_distribution, best_res_out = save_model_check('out', df_check_out_distribution, df['mse_out'].values, best_res_out, df_acc_gen, path_generator)

    return mse_gan_in_distribution, mse_gan_out_distribution, df_acc_gen