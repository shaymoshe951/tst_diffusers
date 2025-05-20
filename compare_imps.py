from scheduler import SchedulerLinear
from tst_my_diff2 import Diffusion
import torch
from matplotlib import pyplot as plt

diff_gt = Diffusion(device='cpu')
diff_tst = SchedulerLinear(device='cpu')

b_gt = diff_gt.beta
b_tst = diff_tst.betatv
print((b_gt-b_tst).norm())

print((diff_gt.alpha_hat - diff_tst.bar_alphatv).norm())

for t in range(1,1000):
    x = torch.randn(5, 1, 32, 32)
    predicted_noise = torch.randn(5, 1, 32, 32)
    noise = torch.randn(5, 1, 32, 32)
    t = [t]
    alpha = diff_gt.alpha[t][:, None, None, None]
    alpha_hat = diff_gt.alpha_hat[t][:, None, None, None]
    beta = diff_gt.beta[t][:, None, None, None]
    fac1_gt = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
    fac2_gt = torch.sqrt(beta) * noise

    t = t[0]
    bar_alphat = diff_tst.bar_alphatv[t]
    alphat = diff_tst.alphatv[t]
    bar_alphat_1 = bar_alphat / alphat
    betat = 1-alphat
    sigmat = torch.sqrt(betat) #(1-bar_alphat_1)/(1-bar_alphat) * betat if t > 1 else 0
    fac1_tst = 1 / torch.sqrt(alphat) * (x - betat * predicted_noise / torch.sqrt(1 - bar_alphat))
    fac2_tst = sigmat * noise
    print(t, (fac1_gt-fac1_tst).norm())
    print(t, (fac2_gt-fac2_tst).norm())

# plt.plot(b_gt)
# plt.plot(b_tst)
# plt.show()

