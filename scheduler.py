import torch

class Scheduler():
    def __init__(self, device='cuda'):
        self.device = device
        self.T = 1000
        self.timesteps = torch.arange(0,self.T).to(device)
        t = self.timesteps
        T = self.T
        s = 0.008
        self.s = s
        # calculate ft
        ftv = torch.cos( (t/T + s)/(1+s) * torch.pi/2).to(device)
        self.bar_alphatv = ftv/ftv[0]
        # plt.plot(t,alphat)
        # plt.show()

    def sample(self, x0):
        t = torch.randint(0, self.T - 1, (x0.shape[0],), device = self.device)
        t = t.view(t.shape[0],1,1,1)
        bar_alphat = self.bar_alphatv[t]
        eps_xt_t = torch.randn(x0.shape, device=self.device)
        xt = torch.sqrt(bar_alphat) * x0 + torch.sqrt(1-bar_alphat) * eps_xt_t
        return xt, eps_xt_t, t

    def step_back(self, xt, eps_th_est, t):
        bar_alphat = self.bar_alphatv[t]
        bar_alphat_1 = self.bar_alphatv[t-1] if t > 1 else 1
        alphat = bar_alphat/bar_alphat_1
        betat = 1-alphat
        sigmat = (1-bar_alphat_1)/(1-bar_alphat) * betat if t > 1 else 0
        noise_t = torch.randn_like(xt, device=self.device)
        x_prev = 1/torch.sqrt(alphat)*( xt - betat*eps_th_est/torch.sqrt(1-bar_alphat) ) + sigmat * noise_t
        return x_prev

class SchedulerLinear():
    def __init__(self, device='cuda'):
        self.device = device
        self.T = 1000
        self.timesteps = torch.arange(0,self.T).to(device)
        t = self.timesteps
        T = self.T
        self.betatv = torch.linspace(0.0001, 0.02, T).to(device)
        self.alphatv = 1-self.betatv
        self.bar_alphatv = torch.cumprod(self.alphatv, dim=0)
        # plt.plot(t,alphat)
        # plt.show()

    def sample(self, x0):
        t = torch.randint(1, self.T, (x0.shape[0],), device = self.device)
        t = t.view(t.shape[0],1,1,1)
        bar_alphat = self.bar_alphatv[t]
        eps_xt_t = torch.randn(x0.shape, device=self.device)
        xt = torch.sqrt(bar_alphat) * x0 + torch.sqrt(1-bar_alphat) * eps_xt_t
        return xt, eps_xt_t, t

    def step_back(self, xt, eps_th_est, t):
        bar_alphat = self.bar_alphatv[t]
        alphat = self.alphatv[t]
        bar_alphat_1 = bar_alphat / alphat
        betat = 1-alphat
        sigmat = (1-bar_alphat_1)/(1-bar_alphat) * betat if t > 1 else 0
        noise_t = torch.randn_like(xt, device=self.device)
        x_prev = 1/torch.sqrt(alphat)*( xt - betat*eps_th_est/torch.sqrt(1-bar_alphat) ) + sigmat * noise_t
        return x_prev


# sc = Scheduler()
# y = sc.get_step(torch.zeros((4,5)),10)
