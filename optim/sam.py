import torch


class SAM_ascent(torch.optim.Optimizer):
    def __init__(self, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho=rho
        defaults = dict(rho=rho, **kwargs)
        super(SAM_ascent, self).__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer
        # self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def step(self):
        # grad_norm = self._grad_norm()
        # print(grad_norm)
        for group in self.param_groups:
            # scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad # * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class SAM_descent(torch.optim.Optimizer):
    def __init__(self, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM_descent, self).__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update