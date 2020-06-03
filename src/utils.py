import torch
import shutil
import os

def save_ckp(state, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True) 
    f_path = os.path.join(checkpoint_dir,'checkpoint.pth')
    torch.save(state, f_path)
    print(f'chekpoints torch file was saved in {f_path}')

def load_ckp_gan(checkpoint_path, generator, discriminator, optimizer_G, optimizer_D):
    checkpoint = torch.load(os.path.join(checkpoint_path, 'checkpoint.pth'))
    generator.load_state_dict(checkpoint['state_dict_generator'])
    discriminator.load_state_dict(checkpoint['state_dict_discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    return generator, discriminator, optimizer_G, optimizer_D, checkpoint['epoch'], checkpoint['df_acc_gen'], checkpoint['best_res_in'], checkpoint['best_res_out']

def load_ckp_forward(checkpoint_path, forward, optimizer, lr_sched):
    checkpoint = torch.load(os.path.join(checkpoint_path, 'checkpoint.pth'))
    forward.load_state_dict(checkpoint['state_dict_forward'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_sched.load_state_dict(checkpoint['lr_sched'])
    return forward, optimizer, lr_sched, checkpoint['epoch'], checkpoint['df_acc_final_in'], checkpoint['df_acc_final_out'], checkpoint['valid_loss_max']