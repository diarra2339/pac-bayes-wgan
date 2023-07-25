import torch
from torch.optim import Adam

from tqdm import tqdm

from utils import get_device, Average
from history import History


class BaseWGAN:
    def __init__(self, generator, critic, prob_gen):
        self.generator = generator
        self.critic = critic
        self.prob_gen = prob_gen

        self.gen_optimizer = None
        self.critic_optimizer = None
        self.train_params = None
        self.history = History()

    def _prep_train(self, lr_g, lr_d):
        self.generator = self.generator.to(get_device())
        self.critic = self.critic.to(get_device())

        self.gen_optimizer = Adam(params=self.generator.parameters(), lr=lr_g, betas=(0.7, 0.999))
        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=lr_d, betas=(0.7, 0.999))

    def _critic_step(self, real_batch):
        raise NotImplementedError

    def _gen_step(self, batch_size):
        latent_batch = torch.randn(batch_size, self.generator.latent_dim).to(get_device())
        fake_images = self.generator(latent_batch)

        fake_logits = self.critic(fake_images)

        loss_gen = - fake_logits.mean().view(-1)  # shape becomes [1], instead of []
        return loss_gen

    def train_model(self, train_loader, epochs, critic_steps, kl_coeff=1, lr_g=1e-4, lr_d=1e-4):
        assert critic_steps > 0
        self._prep_train(lr_g=lr_g, lr_d=lr_d)

        for epoch in range(1, epochs + 1):
            print('Epoch {} ...'.format(epoch))
            critic_losses, gen_losses, grad_penalties, kl_losses = Average(), Average(), Average(), Average()
            for real_batch, _ in tqdm(train_loader):
                batch_size = real_batch.shape[0]
                real_batch = real_batch.to(get_device())

                # train the critic for k steps
                self.generator.eval()
                self.critic.train()
                for _ in range(critic_steps):
                    self.critic_optimizer.zero_grad()
                    loss_critic = self._critic_step(real_batch=real_batch)

                    loss_critic.backward()
                    self.critic_optimizer.step()
                critic_losses.add(loss_critic)

                # train the generator once
                self.critic.eval()
                self.generator.train()
                self.gen_optimizer.zero_grad()
                loss_gen = self._gen_step(batch_size=batch_size)
                if self.prob_gen:
                    kl_loss = self.generator.kl_div
                    loss_gen += kl_coeff * kl_loss
                    kl_losses.add(kl_loss)

                loss_gen.backward()
                self.gen_optimizer.step()

                gen_losses.add(loss_gen)

            # End of epoch
            critic_loss, gen_loss, grad_penalty = critic_losses.compute(), gen_losses.compute(), grad_penalties.compute()
            kl_loss = kl_losses.compute() if self.prob_gen else None
            display = 'critic_loss: {:.3f}; gen_loss: {:.3f}'.format(
                critic_losses.compute(), gen_losses.compute()
            )
            logs = {'critic-loss': critic_loss,
                    'gen_loss': gen_losses.compute(),
                    }
            if self.prob_gen:
                display += '; kl_loss: {:.3f}'.format(kl_loss)
                logs['kl_div'] = kl_loss

            print(display)
            self.history.save(logs)

        print('End of training!')

    def generate_samples(self, num_samples, sample=True):
        latent_batch = torch.randn(num_samples, self.generator.latent_dim).to(get_device())
        samples = self.generator(latent_batch, sample) if self.prob_gen else self.generator(latent_batch)
        return samples.detach()


class BjorckWGAN(BaseWGAN):
    def __init__(self, generator, critic, prob_gen):
        super(BjorckWGAN, self).__init__(generator=generator, critic=critic, prob_gen=prob_gen)

    def _critic_step(self, real_batch):
        batch_size = real_batch.shape[0]
        latent_batch = torch.randn(batch_size, self.generator.latent_dim).to(get_device())
        fake_batch = self.generator(latent_batch)

        real_logits = self.critic(real_batch)
        fake_logits = self.critic(fake_batch)

        loss_critic = fake_logits.mean() - real_logits.mean()

        return loss_critic

