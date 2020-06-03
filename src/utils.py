import torch
import shutil

def save_ckp(state, checkpoint_dir):
    f_path = checkpoint_dir / 'checkpoint.pt'
    torch.save(state, f_path)

def load_ckp_gan(checkpoint_fpath, generator, discriminator, optimizer_G, optimizer_D):
    checkpoint = torch.load(checkpoint_fpath)
    generator.load_state_dict(checkpoint['state_dict_generator'])
    discriminator.load_state_dict(checkpoint['state_dict_discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    df_acc_gen = checkpoint['df_acc_gen']
    return generator, discriminator, optimizer_G, optimizer_D, checkpoint['epoch'], df_acc_gen

def load_ckp_forward(checkpoint_fpath, forward, optimizer, lr_sched):
    checkpoint = torch.load(checkpoint_fpath)
    forward.load_state_dict(checkpoint['state_dict_forward'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_sched.load_state_dict(checkpoint['lr_sched'])
    #df_acc_gen = checkpoint['df_acc_gen']
    return generator, discriminator, optimizer_G, optimizer_D, checkpoint['epoch'], df_acc_gen