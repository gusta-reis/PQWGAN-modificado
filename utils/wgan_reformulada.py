import torch
import torch.autograd as autograd

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    # Assegurar que o número de amostras seja o mesmo
    batch_size = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:batch_size]
    fake_samples = fake_samples[:batch_size]

    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(real_samples)

    # Interpolação entre amostras reais e falsas
    interpolated_images = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)

    # Passa as imagens interpoladas pelo crítico
    critic_interpolated = critic(interpolated_images)

    # Calcula gradientes das saídas do crítico em relação às imagens interpoladas
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Calcula a penalidade do gradiente
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    gradient_penalty += 1e-8
    
    return gradient_penalty