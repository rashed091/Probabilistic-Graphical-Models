import pyro
import torch.distributions as tdist


pyro.set_rng_seed(1234)

# Primitive Stochastic Functions

loc = 0
scale = 1.0
normal = tdist.Normal(loc, scale)
x = normal.rsample()

print(x)
print("log prob", normal.log_prob(x))  # score the sample from N(0,1)

# A simple model
def weather():
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()


for _ in range(3):
    print(weather())
