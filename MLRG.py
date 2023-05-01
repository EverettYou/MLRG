import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


''' Meaning of dimensions
    n - size of a single G-spin (number of group elements in G)
    k - number of non-trivial conjugacy class in G
    d - point group representation dimension of spin
    c - number of non-trivial coupling types under D4h
    L - number of sites (spins)
    M - number of samples
'''

# Sn group non-trivial class projectors
Sn_dat = {2: torch.tensor([[[0], [1]], [[1], [0]]], dtype=torch.float),
          3: torch.tensor([[[0, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]], [[1, 0], [0, 0], [0, 1], [1, 0], [1, 0], [0, 1]], [[1, 0], [0, 1], [0, 0], [1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [1, 0], [0, 0], [0, 1], [1, 0]], [[0, 1], [1, 0], [1, 0], [0, 1], [0, 0], [1, 0]], [[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 0]]], dtype=torch.float)}

# D4h group irreducible representations
rep_dim = {'A1': 1, 'A2': 1, 'B1': 1, 'B2': 1, 'E': 2}
rep_C4  = {'A1': torch.tensor([[1.]]),
           'A2': torch.tensor([[1.]]),
           'B1': torch.tensor([[-1.]]),
           'B2': torch.tensor([[-1.]]),
           'E' : torch.tensor([[0.,-1.],[1.,0.]])}
rep_sig = {'A1': torch.tensor([[1.]]),
           'A2': torch.tensor([[-1.]]),
           'B1': torch.tensor([[1.]]),
           'B2': torch.tensor([[-1.]]),
           'E' : torch.tensor([[1.,0.],[0.,-1.]])}

# bi-adjacency matrices
biadj_dat = {'square': torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]], dtype=torch.float),
             'cross': torch.tensor([[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 1]]], dtype=torch.float)}

class Bond(torch.nn.Module):
    ''' Directed bond model - process internal and point group symmetry data

        Parameters:
            invar_ten (Tensor):  G-invariant tensor of shape (n, n, k) 
            rep_src (list of str): D4h representation of source spin 
            rep_tgt (list of str): D4h representation of target spin 
                                    (set to rep_src if not specified) 
    '''
    def __init__(self, invar_ten, rep_src, rep_tgt=None):
        super().__init__()
        self.register_buffer('invar_ten', invar_ten) # (n, n, k)
        self.n, _, self.k = self.invar_ten.shape
        self.rep_src = self.make_rep(rep_src)
        self.rep_tgt = self.make_rep(rep_tgt) if rep_tgt is not None else self.rep_src
        self.d_src = sum(rep_dim[rep] for rep in self.rep_src)
        self.d_tgt = sum(rep_dim[rep] for rep in self.rep_tgt)
        self.register_buffer('fusion_ten', self.get_fusion_ten()) # (L_src d_src, L_tgt d_tgt, l)
        self.c = self.fusion_ten.shape[-1]
        self.Jdim = self.c * self.k # total number of coupling constants

    def extra_repr(self):
        return f'{self.n}, {self.rep_src} -> {self.rep_tgt}'

    @staticmethod
    def make_rep(rep):
        if isinstance(rep, (tuple, list)):
            if len(rep) > 0:
                return list(rep)
            else:
                raise ValueError('rep can not be empty.')
        elif isinstance(rep, str):
            return [rep]
        else:
            raise ValueError(f'rep should be a string or a list of strings, received {rep}')

    def get_fusion_ten(self):
        ''' D4h representation fusion tensor '''
        # construct reflection matrix
        sig_src = torch.block_diag(*[rep_sig[rep] for rep in self.rep_src]) # (d_src, d_src)
        sig_tgt = torch.block_diag(*[rep_sig[rep] for rep in self.rep_tgt]) # (d_tgt, d_tgt)
        sig = torch.kron(sig_src, sig_tgt) # (d_src d_tgt, d_scr d_tgt)
        # construct reflection symmetry projector
        proj = sig - torch.eye(sig.shape[0]).to(sig) # (d_src d_tgt, d_scr d_tgt)
        # find null space
        q, r = torch.linalg.qr(proj, mode='complete') # (d_src d_tgt, d_scr d_tgt)
        msk = torch.diag(r) == 0 # (d_scr d_tgt, )
        nullspace = q[:, msk] # (d_src d_tgt, c)
        # construct C4 rotations
        C4_src = torch.block_diag(*[rep_C4[rep] for rep in self.rep_src]) # (d_src, d_src)
        C4_tgt = torch.block_diag(*[rep_C4[rep] for rep in self.rep_tgt]) # (d_tgt, d_tgt)
        C4 = torch.kron(C4_src, C4_tgt) # (d_src d_tgt, d_scr d_tgt)
        fusion_ten = [nullspace]
        for _ in range(3):
            nullspace = C4.mm(nullspace) # (d_src d_tgt, c)
            fusion_ten.append(nullspace)
        fusion_ten = torch.stack(fusion_ten).view(4, self.d_src, self.d_tgt, -1) # (4, d_src, d_tgt, c)
        return fusion_ten

class EquivariantRBM(torch.nn.Module):
    ''' Equivariant Ristricted Boltzmann Machine 

        Parameters:
            biadj (Tensor): bi-adjacency matrix of shape (L_src, L_tgt, 4)
            bond (Bond): directed bond model
    '''
    def __init__(self, biadj, bond):
        super().__init__()
        self.biadj = biadj
        self.L_src, self.L_tgt, _ = self.biadj.shape
        self.bond = bond
        self.Ld_src, self.Ld_tgt = self.L_src * self.bond.d_src, self.L_tgt * self.bond.d_tgt
        self.register_buffer('coupling_ten', self.get_coupling_ten()) # (L_src d_src, L_tgt d_tgt, c)
        self.clear_cache()

    @property
    def device(self):
        return self.coupling_ten.device

    def extra_repr(self):
        return f'spins: {self.Ld_src} -> {self.Ld_tgt}, freedom: {self.bond.Jdim},'

    def get_coupling_ten(self):
        ''' Bipartite graph model coupling tensor '''
        coupling_ten = torch.einsum(self.biadj, [0,1,2], self.bond.fusion_ten, [2,3,4,5], [0,3,1,4,5]) # (L_src, d_src, L_tgt, d_tgt, c)
        return coupling_ten.reshape(self.Ld_src, self.Ld_tgt, self.bond.c) # (Ld_src, Ld_tgt, c)

    def clear_cache(self):
        ''' clear cache variables '''
        self._J = None
        self._kernel = None
        self._weight = None

    def set_J(self, J=None):
        ''' set coupling constants J 

            Input:
                J (Tensor): coupling parameters (..., Jdim)
                [otherwise, each component in J will be mapped to
                 a (c,k) coupling matrix with its value written to (0,0)
                 component of the coupling matrix]
        '''
        if J is None:
            J = torch.nn.Parameter(torch.randn(self.bond.Jdim, device=self.device))
        else:
            if not isinstance(J, torch.Tensor):
                J = torch.tensor(J)
            if J.dim() == 0:
                J = J.unsqueeze(-1)
            if J.shape[-1] != self.bond.Jdim:
                raise ValueError(f'The last dimension of J {J.shape[-1]} does not match Jdim {self.bond.Jdim}.')
        self.clear_cache()
        self._J = J

    @property
    def J(self):
        ''' coupling constants (J) viewed as (..., c, k) '''
        return self._J.view(self._J.shape[:-1] + (self.bond.c, self.bond.k))

    @property
    def kernel(self):
        ''' energy model kernel '''
        if self._kernel is None:
            self._kernel = torch.einsum(self.J, [...,0,1], self.coupling_ten, [2,3,0], self.bond.invar_ten, [4,5,1], [...,2,4,3,5]) # (..., Ld_src, n, Ld_tgt, n)
        return self._kernel # (..., Ld_src, n, Ld_tgt, n)

    ''' Common tensors:
            src (Tensor): source spin configruations (..., M, Ld_src, n)
            tgt (Tensor): target spin configruations (..., M, Ld_tgt, n)
            energy (Tensor): energy of spin configruations (..., M)
            cdloss (Tensor): contrastive divergence loss (..., M)
            weight (Tensor): partition weight tensor (..., *[n^d]*L)
        Common parameters:
            samples (int): number of samples to drawn together
            steps (int): number of iterations for Gibbs sampling to converge 
    '''

    def potential_tgt(self, src):
        ''' compute potential energy of target spins given source spins '''
        return torch.einsum(self.kernel, [...,0,1,2,3], src, [...,4,0,1], [...,4,2,3]) # (..., M, Ld_tgt, n)

    def potential_src(self, tgt):
        ''' compute potential energy of source spins given target spins '''
        return torch.einsum(self.kernel, [...,0,1,2,3], tgt, [...,4,2,3], [...,4,0,1]) # (..., M, Ld_src, n)

    def energy(self, src, tgt):
        ''' energy function '''
        return torch.sum(src * self.potential_src(tgt), dim=(-2,-1)) # (..., M)

    def one_hot_sample(self, potential):
        ''' given potential, sample spin configurations, returns one-hot representation
            
            Input:
                potential (Tensor): negative logits (..., M, Ld, n)

            Output:
                out (Tensor): spin configurations (..., M, Ld, n)
        '''
        indx = torch.distributions.Categorical(logits= - potential).sample() # (..., M, Ld)
        return torch.nn.functional.one_hot(indx, num_classes=self.bond.n).to(torch.float) # (..., M, Ld, n)

    def sample_src(self, samples=1, steps=5):
        ''' Gibbs sampling source configurations '''
        batch_shape = self.J.shape[:-2]
        indx = torch.randint(self.bond.n, batch_shape + (samples, self.Ld_src), device=self.J.device) # (..., M, Ld_src)
        src = torch.nn.functional.one_hot(indx, num_classes=self.bond.n).to(torch.float) # (..., M, Ld_src, n)
        for _ in range(steps):
            tgt = self.one_hot_sample(self.potential_tgt(src)) # (..., M, Ld_tgt, n)
            src = self.one_hot_sample(self.potential_src(tgt)) # (..., M, Ld_src, n)
        return src

    def sample_tgt(self, samples=1, steps=5):
        ''' Gibbs sampling target configurations '''
        batch_shape = self.J.shape[:-2]
        indx = torch.randint(self.bond.n, batch_shape + (samples, self.Ld_tgt), device=self.J.device) # (..., M, Ld_tgt)
        tgt = torch.nn.functional.one_hot(indx, num_classes=self.bond.n).to(torch.float) # (..., M, Ld_tgt, n)
        for _ in range(steps):
            src = self.one_hot_sample(self.potential_src(tgt)) # (..., M, Ld_src, n)
            tgt = self.one_hot_sample(self.potential_tgt(src)) # (..., M, Ld_tgt, n)
        return tgt

    def cdloss_src(self, src0, steps=1):
        ''' contrastive divergence loss from source configurations '''
        if steps == 1:
            tgt = self.one_hot_sample(self.potential_tgt(src0)) # (..., M, Ld_tgt, n)
            potential_src = self.potential_src(tgt)  # (..., M, Ld_src, n) 
            src = self.one_hot_sample(potential_src) # (..., M, Ld_src, n)
            loss = torch.sum(potential_src * (src0 - src), dim=(-2,-1)) # (..., M)
        else:
            src = src0
            for t in range(steps):
                tgt = self.one_hot_sample(self.potential_tgt(src)) # (..., M, Ld_tgt, n)
                if t == 0:
                    tgt0 = tgt
                src = self.one_hot_sample(self.potential_src(tgt)) # (..., M, Ld_src, n)
            loss = self.energy(src0, tgt0) - self.energy(src, tgt) # (..., M)
        return loss

    def cdloss_tgt(self, tgt0, steps=1):
        ''' contrastive divergence loss from target configurations '''
        if steps == 1:
            src = self.one_hot_sample(self.potential_src(tgt0)) # (..., M, Ld_src, n)
            potential_tgt = self.potential_tgt(src)  # (..., M, Ld_tgt, n)
            tgt = self.one_hot_sample(potential_tgt) # (..., M, Ld_tgt, n)
            loss = torch.sum(potential_tgt * (tgt0 - tgt), dim=(-2,-1)) # (..., M)
        else:
            tgt = tgt0
            for t in range(steps):
                src = self.one_hot_sample(self.potential_src(tgt)) # (..., M, Ld_src, n)
                if t == 0:
                    src0 = src
                tgt = self.one_hot_sample(self.potential_tgt(src)) # (..., M, Ld_tgt, n)
            loss = self.energy(src0, tgt0) - self.energy(src, tgt) # (..., M)
        return loss

    def one_hot_configs(self, Ld):
        ''' enumerate one-hot configurations 

            Input:
                Ld (int): total number of G-spins (L * d)

            Output:
                out (Tensor): spin configurations (n^Ld, Ld, n)
        '''
        indx = torch.arange(self.bond.n, device=self.device) # (n, )
        if Ld == 1:
            indx = indx.unsqueeze(-1) # (n, 1)
        elif Ld > 1:
            indx = torch.cartesian_prod(* [indx] * Ld) # (n^Ld, Ld)
        else:
            raise ValueError(f'Ld value {Ld} is not valid.')
        return torch.nn.functional.one_hot(indx, num_classes=self.bond.n).to(torch.float) # (n^Ld, Ld, n)

    @property
    def weight(self):
        ''' Boltzmann weight of all spins (WARNING: exponential complexity) '''
        if self._weight is None:
            src = self.one_hot_configs(self.Ld_src) # (n^Ld_src, Ld_src, n)
            tgt = self.one_hot_configs(self.Ld_tgt) # (n^Ld_tgt, Ld_tgt, n)
            potential_src = self.potential_src(tgt) # (..., n^Ld_tgt, Ld_src, n)
            energy = torch.einsum(src, [0,2,3], potential_src, [...,1,2,3], [...,0,1]) # (..., n^Ld_src, n^Ld_tgt)
            self._weight = torch.exp(-energy) # (..., n^Ld_src, n^Ld_tgt)
        return self._weight # (..., n^Ld_src, n^Ld_tgt)

    def prob_src(self, split=False):
        ''' source spin configuration probabilities (WARNING: exponential complexity) 

            Input:
                split (bool): if True, partition the probability vector into tensor by sites
        '''
        weight_src = self.weight.sum(-1) # (..., n^Ld_src)
        prob_src = weight_src / weight_src.sum(-1, keepdim=True) # (..., n^Ld_src)
        if split:
            prob_src = prob_src.view(prob_src.shape[:-1]+(self.bond.n ** self.bond.d_src,) * self.L_src) # (..., *[n^d_src]*L_src)
        return prob_src

    def prob_tgt(self, split=False):
        ''' target spin configuration probabilities (WARNING: exponential complexity) 

            Input:
                split (bool): if True, partition the probability vector into tensor by sites
        '''
        weight_tgt = self.weight.sum(-2) # (..., n^Ld_tgt)
        prob_tgt = weight_tgt / weight_tgt.sum(-1, keepdim=True) # (..., n^Ld_tgt)
        if split:
            prob_tgt = prob_tgt.view(prob_tgt.shape[:-1]+(self.bond.n ** self.bond.d_tgt,) * self.L_tgt) # (..., *[n^d_tgt]*L_tgt)
        return prob_tgt

def Tvals(T, power=1):
    ''' compute transfer matrix eigenvalues '''
    assert max(T.shape[-4:])**power <= 1024, f'power={power} is too large.'
    TT = T
    for _ in range(power-1):
        TT = torch.einsum(TT, [...,0,2,3,6], T, [...,1,6,4,5], [...,0,1,2,3,4,5])
        TT = TT.reshape(TT.shape[:-6]+(TT.shape[-6]*TT.shape[-5],TT.shape[-4],TT.shape[-3]*TT.shape[-2],TT.shape[-1]))
        TT = TT / torch.sum(TT, dim=(-4,-3,-2,-1), keepdim=True)
    M = torch.einsum(TT, [...,0,2,1,2], [...,0,1])
    MM = M
    for _ in range(power-1):
        MM = torch.matmul(MM, M)
        MM = MM / torch.sum(MM, dim=(-2,-1), keepdim=True)
    vals = torch.linalg.eigvalsh(MM).abs()
    return vals

def GSD(T):
    ''' compute ground state degeneracy '''
    lamXi = torch.einsum(T, [...,0,1,0,1], [...])
    lam2Xi = torch.einsum(T, [...,0,1,2,1], T, [...,2,3,0,3], [...])
    return lamXi**2/lam2Xi

class RGMonotone(torch.nn.Module):
    ''' RG Monotone network (feed-forward) 

        Parameters:
            dims (list of int): dimensions from input to output '''
    def __init__(self, Jdim, hdims=[]):
        super().__init__()
        layers = []
        self.Jdim = Jdim
        dim_in = self.Jdim
        for dim_out in hdims:
            layers.append(torch.nn.Linear(dim_in, dim_out))
            layers.append(torch.nn.LayerNorm(dim_out))
            layers.append(torch.nn.Tanh())
            dim_in = dim_out
        layers.append(torch.nn.Linear(dim_in, 1)) # map to a scalar
        self.ffn = torch.nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, J):
        ''' compute RG monotone 

            Input:
                J (Tensor): coupling constants of shape (..., Jdim) 

            Output:
                C (Tensor): RG monotone (...,) '''
        C = self.ffn(J)
        C -= self.ffn(torch.zeros_like(J)).detach() # fix global shift
        return C.squeeze(-1)

    @torch.no_grad()
    def C(self, J):
        ''' RG monotone C(J) (no grad) '''
        return self(J) # (...,)

    def gradC(self, J):
        ''' gradient of RG monotone dC/dJ'''
        with torch.set_grad_enabled(True):
            J.requires_grad_(True)
            C = self(J)
            gradC, *_ = torch.autograd.grad(C.sum(), J, create_graph=True)
        return gradC # (..., Jdim)

    def RGforward(self, J):
        ''' forward RG flow from J '''
        return J - self.gradC(J) # (..., Jdim)
    
    def RGbackward(self, J):
        ''' backward RG flow from J '''
        return J + self.gradC(J) # (..., Jdim)
    
    @torch.no_grad()
    def Newton_step(self, J, step_size=0.01): 
        # torch.autograd.functional.jacobian cannot do batch-wise
        assert J.numel() == self.Jdim, f'size of J {J.numel()} does not match Jdim {self.Jdim}.'
        J = J.view(-1)
        Jac = torch.autograd.functional.jacobian(self.gradC, J)
        J = J - step_size * torch.linalg.inv(Jac) @ self.gradC(J)
        return J

class HMCSampler():
    def __init__(self, energy):
        self.energy = energy
        
    def grad_energy(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            total_energy = self.energy(x).sum()
            grad_energy = torch.autograd.grad(total_energy, x)[0]
        x.detach()
        return grad_energy
    
    def leap_frog(self, x0, p0, dt=0.01, traj_len=32):
        with torch.no_grad():
            x, p = x0, p0
            p = p - 0.5 * dt * self.grad_energy(x)
            x = x + dt * p
            for t in range(traj_len):
                p = p - dt * self.grad_energy(x)
                x = x + dt * p
            p = p - 0.5 * dt * self.grad_energy(x)
        return x, p
    
    def hamiltonian(self, x, p):
        V = self.energy(x)
        K = (p ** 2).sum(-1) / 2
        return K + V
    
    def step(self, x0, **kwargs):
        p0 = torch.randn_like(x0)
        H0 = self.hamiltonian(x0, p0)
        x, p = self.leap_frog(x0, p0, **kwargs)
        H = self.hamiltonian(x, p)
        prob_accept = torch.exp(H0 - H)
        mask = prob_accept > torch.rand_like(prob_accept)
        x = torch.where(mask[...,None], x, x0)
        return x
    
    def update(self, x, steps=1, **kwargs):
        for _ in range(steps):
            x = self.step(x, **kwargs)
        return x

class MLRG(torch.nn.Module):
    ''' Machine-Learning Renormalization Group'''
    def __init__(self, invar_ten, rep, hdims=None):
        super().__init__()
        bond = Bond(invar_ten, rep)
        self.teacher =  EquivariantRBM(biadj_dat['square'], bond)
        self.student =  EquivariantRBM(biadj_dat['cross' ], bond)
        if hdims is None:
            hdims = [8*bond.Jdim,4*bond.Jdim]
        self.moderator = RGMonotone(bond.Jdim, hdims=hdims)

    def propose(self, Jtch=None, beta=1., lamb=0., mu=0., batch=16, steps=1):
        ''' propose random coupling constants J 

            Input:
                beta (float or Tensor): inverse temperature. If Tensor: shape (batch,)
                steps (int): HMC steps
        '''
        if Jtch is None:
            Jtch = torch.randn((batch, self.moderator.Jdim), device=self.moderator.device)
        sampler = HMCSampler(energy=lambda J0: (beta * self.moderator.gradC(J0).norm(dim=-1) + lamb * self.moderator(J0) + mu * J0.norm(dim=-1)))
        Jtch = sampler.update(Jtch, steps=steps).detach()
        return Jtch

    def loss(self, Jtch, samples=512, gibbssteps=5, cdsteps=1):
        ''' loss function '''
        self.teacher.set_J(Jtch)
        Jstd = self.moderator.RGforward(Jtch)
        self.student.set_J(Jstd)
        data = self.teacher.sample_tgt(samples=samples, steps=gibbssteps)
        loss = self.student.cdloss_src(data, steps=cdsteps).mean(-1)
        return loss.mean(-1)

        












