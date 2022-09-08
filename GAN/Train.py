import torch
import torch.nn.functional as f
import pandas as pd
from NN_models import *
from utils import *
import matplotlib.pyplot as plt
import wandb

wandb.init(project="GAN", config={"hyper":"paramet"})

device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_data)

    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp

    return crit_loss


display_step = 50



def train(df, epochs=500, batch_size=32):
    ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test = prepare_data(df, batch_size)

    generator = Generator(input_dim, continuous_columns, discrete_columns).to(device)
    critic = Critic(input_dim).to(device)
    #second_critic = FairLossFunc(S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # generator = torch.load("Results/GeneratorAllData.pth")
    # critic = torch.load("Results/CriticAllDatapth")
    # wandb.watch(critic,log_freq=100)
    # wandb.watch(generator,log_freq=100)

    
    print("model = ", gen_optimizer)
    print("optimizer = ", gen_optimizer)



    # loss = nn.BCELoss()
    critic_losses = []
    generator_losses = []
    cur_step = 0
    for i in range(epochs):
        # j = 0
        print("epoch {}".format(i + 1))
        ############################
        for data in train_dl:
            data[0] = data[0].to(device)
            # j += 1
            loss_of_epoch_G = 0
            loss_of_epoch_D = 0
            crit_repeat = 4
            mean_iteration_critic_loss = 0
            for k in range(crit_repeat):
                # training the critic
                crit_optimizer.zero_grad()
                fake_noise = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake = generator(fake_noise)

                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(data[0])

                epsilon = torch.rand(batch_size, input_dim, device=device, requires_grad=True)
                gradient = get_gradient(critic, data[0], fake.detach(), epsilon)
                gp = gradient_penalty(gradient)

                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10)

                mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()
            #############################
            if cur_step > 50:
                critic_losses += [mean_iteration_critic_loss]
                # wandb.log({"Critic_loss": critic_losses})

            #############################

            gen_optimizer.zero_grad()
            fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
            fake_2 = generator(fake_noise_2)
            crit_fake_pred = critic(fake_2)

            gen_loss = get_gen_loss(crit_fake_pred)
            gen_loss.backward()

                # Update the weights
            gen_optimizer.step()

    
            
            # Keep track of the average generator loss
            #################################
            if cur_step > 50:
                generator_losses += [gen_loss.item()]
                # wandb.log({"generator_loss": generator_losses})
                
                

        if cur_step % display_step == 0 and cur_step > 0:
            

            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            # print("Step {}: Generator loss: {}, critic loss: {}".format(cur_step, gen_mean, crit_mean))
        step_bins = 20
        num_examples = (len(generator_losses) // step_bins) * step_bins
        plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                )
        plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                )
        plt.savefig('Results/GanResults.png',bbox_inches = 'tight')
        plt.show()

	    
        cur_step += 1
        torch.save(generator, "Results/GeneratorAllclassestest.pth")
        torch.save(critic, "Results/CriticAllclassestest.pth")
    return generator, critic, ohe, scaler, data_train, data_test, input_dim





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(1))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
print(device)


def train_plot(df, epochs, batchsize):
    generator, critic, ohe, scaler, data_train, data_test, input_dim = train(df, epochs, batchsize)
    return generator, critic, ohe, scaler, data_train, data_test, input_dim

size_of_fake_data=200000
num_epochs=70
batch_size=64

df=pd.read_csv("combined.csv")
fe=pd.read_csv("DDoS_Functional_Features.csv")
# df=df.loc[df['Label']==1]s
# df=df.drop("Label",axis=1)
# df,n=drop_function(df,fe)

fake_name="Results/generatedAllclassestest.csv"
generator, critic, ohe, scaler, data_train, data_test, input_dim = train_plot(df,num_epochs, batch_size)

fake_numpy_array = generator(torch.randn(size=(size_of_fake_data, input_dim), device=device)).cpu().detach().numpy()
fake_df = get_original_data(fake_numpy_array, df, ohe, scaler)
fake_df = fake_df[df.columns]
fake_df.to_csv(fake_name, index=False) 

