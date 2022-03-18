from typing import List, Tuple, Sequence, Union, Any, Dict
import math
import os
import numpy as np

# random seed
np.random.seed(12345)

def set_random_seed(seed: int):
    np.random.seed(seed)


class Tensor:
    """ 
    The Tensor class is a Tensor data structure, with the underlying data stored in a multidimensional array. This class is very similar to torch.Tensor.

    Tensor.data is the field that contains the main data for this tensor, this field is a np.ndarray. The updates of the parameters should be directly changing this data.

    Tensor.grad is the field for storing the gradient for this tensor. There can be three types of values for this field:
        (1) None: which denotes zero gradient.
        (2) np.ndarray: which should be the same size as the Tensor.data, denoting dense gradients.
        (3) Dict[int, np.ndarray]: which is a simple simulation of sparse gradients for 2D matrices (embeddings). The key int denotes the index into the first dimension, while the value is a np.ndarray which shape is Tensor.data.shape[1], denoting the gradient for the column slice according to the index.

    Tensor.op, which is an Op (see below) that generates this Tensor. If None, then mostly not calculated but inputted.

    """

    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data
        self.grad: Union[Dict[int, np.ndarray], np.ndarray] = None
        self.op: Op = None

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"T{self.shape}: {self.data}"

    def accumulate_grad(self, g: np.ndarray) -> None:
        """
        accumulate_grad accepts one dense np.ndarray and accumulate to the Tensor's dense gradients (np.ndarray).

        """

        ### Add your implementation below and comment the following line.
        # raise NotImplementedError
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad +=g

    def accumulate_grad_sparse(self, gs: List[Tuple[int, np.ndarray]]) -> None:
        """
        accumulate_grad_sparse accepts a list of (index, np.ndarray) and accumulates them to the Tensor's simulated sparase gradients (dict). only for D2 lookup matrix!

        """

        ### Add your implementation below and comment the following line.
        # raise NotImplementedError
        if self.grad is None:
            self.grad = {}
        for each in gs:
            if each[0] in self.grad.keys():
                self.grad[each[0]] += each[1]
            else:
                self.grad[each[0]] = each[1].copy()

    def get_dense_grad(self):
        """
        convert from simulated sparse gradients to dense ones
        """
        ret = np.zeros_like(self.data)
        if self.grad is not None:
            if isinstance(self.grad, dict):
                for widx, arr in self.grad.items():
                    ret[widx] += arr
            else:
                ret = self.grad
        return ret

    def __add__(self, other: 'Tensor'):
        return OpAdd().full_forward(self, other)

    def __sub__(self, other: 'Tensor'):
        return OpAdd().full_forward(self, other, alpha_b=-1.)

    def __mul__(self, other: Union[int, float]):
        assert isinstance(other, (int, float)), "currently only support scalar __mul__"
        return OpAdd().full_forward(self, b=None, alpha_a=float(other))


class Parameter(Tensor):
    """
    Parameter is a simple sub-class of Tensor, denoting persistent model parameters.
    
    """

    def __init__(self, data: np.ndarray):
        super().__init__(data)

    @classmethod
    def from_tensor(cls, tensor: Tensor):
        return Parameter(tensor.data)  # currently simply steal its data


def astensor(t):
    """
    shortcut for create tensor
    """
    return t if isinstance(t, Tensor) else Tensor(np.asarray(t))


class Op:
    """
    Operation class implements an operation that is part of a ComputationGraph.

    Op.ctx: this field is a data field that is populated during the forward operation to store all the relevant values (input, output, intermediate) that must be used in backward to calculate gradients. We provide two helper methods Op.store_ctx() and Op.get_ctx() to do these, but please feel free to store things in your own way, we will not check Op.ctx.

    Op.forward() and Op.backward(): these are the forward and backward methods calculating the operation itself and its gradient.

    Op.full_forward(): This is a simple wrapper for the actual forward, adding only one thing to make it convenient, recording the Tensor.op for the outputted Tensor so that you do not need to add this in forward.

    """

    def __init__(self):
        self.ctx: Dict[str, Union[Tensor, Any]] = {}  # store intermediate tensors or other values
        self.idx: int = None  # idx in the computation graph
        ComputationGraph.get_cg().reg_op(self)  # register into the computation graph

    # store intermediate results for usage in backward
    def store_ctx(self, ctx: Dict = None, **kwargs):
        if ctx is not None:
            self.ctx.update(ctx)
        self.ctx.update(kwargs)

    # get stored ctx values
    def get_ctx(self, *names: str):
        return [self.ctx.get(n) for n in names]

    # full forward, forwarding plus set output op
    def full_forward(self, *args, **kwargs):
        rets = self.forward(*args, **kwargs)
        # -- store op for outputs
        outputs = []
        if isinstance(rets, Tensor):
            outputs.append(rets)  # single return
        elif isinstance(rets, (list, tuple)):  # note: currently only support list or tuple!!
            outputs.extend([z for z in rets if isinstance(z, Tensor)])
        for t in outputs:
            assert t.op is None, "Error: should only have one op!!"
            t.op = self
        # --
        return rets

    # forward the operation
    def forward(self, *args, **kwargs):
        # you will override this function in a subclass below
        raise NotImplementedError()

    # backward with the pre-stored tensors
    def backward(self):
        # you will override this function in a subclass below
        raise NotImplementedError()


### Backpropable Operations
"""
The remaining Op* are all sub-classes of Op and denotes a specific function. 
We provide some operations: OpDropout, OpSum, OpRelu, OpLogloss, OpAdd.

Take OpDropout as an example, here we implement the inverted dropout, which scales values by 1/(1-drop) in forward. In forward, (if training), we obtain a mask using np.random and multiply the input by this. All the intermediate values (including input and output) are stored using store_ctx. In backward, we obtain the graident of the output Tensor by retriving previous stored values. Then the calcualted gradients are assigned to the input Tensor by accumulate_grad.

You need to implement others: OpLookup, OpDot, OpTanh, OpMax, (optional) OpAvg.
"""

class OpDropout(Op):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, drop: float, is_training: bool):
        if is_training:
            arr_mask = np.random.binomial(1, 1.-drop, x.shape) * (1./(1-drop))
            arr_drop = (x.data * arr_mask)
            t_drop = Tensor(arr_drop)
        else:
            arr_mask = 1.
            t_drop = Tensor(x.data)  # note: here copy things to make it consistent!
        self.store_ctx(is_training=is_training, x=x, arr_mask=arr_mask, t_drop=t_drop)
        return t_drop

    def backward(self):
        is_training, x, arr_mask, t_drop = self.get_ctx('is_training', 'x', 'arr_mask', 't_drop')
        if not is_training:
            pass
            # print("Warn: Should not backward if not in training??")
        if t_drop.grad is not None:
            x.accumulate_grad(arr_mask * t_drop.grad)

class OpSum(Op):
    def __init__(self):
        super().__init__()

    # [..., K, ...] -> [..., ...]
    def forward(self, emb: Tensor, axis: int):
        reduce_size = emb.data.shape[axis]
        arr_sum = emb.data.sum(axis=axis)
        t_sum = Tensor(arr_sum)
        self.store_ctx(emb=emb, t_sum=t_sum, axis=axis, reduce_size=reduce_size)
        return t_sum

    def backward(self):
        emb, t_sum, axis, reduce_size = self.get_ctx('emb', 't_sum', 'axis', 'reduce_size')
        if t_sum.grad is not None:
            g0 = np.expand_dims(t_sum.grad, axis)
            g = np.repeat(g0, reduce_size, axis=axis)
            emb.accumulate_grad(g)
        # --

class OpRelu(Op):
    def __init__(self):
        super().__init__()

    # [N] -> [N]
    def forward(self, t: Tensor):
        arr_relu = t.data  # [N]
        arr_relu[arr_relu < 0.0] = 0.0
        t_relu = Tensor(arr_relu)
        self.store_ctx(t=t, t_relu=t_relu, arr_relu=arr_relu)
        return t_relu

    def backward(self):
        t, t_relu, arr_relu = self.get_ctx('t', 't_relu', 'arr_relu')
        if t_relu.grad is not None:
            grad_t = np.where(arr_relu > 0.0, 1.0, 0.0) * t_relu.grad  # [N]
            t.accumulate_grad(grad_t)
        # --

class OpLogloss(Op):
    def __init__(self):
        super().__init__()

    # [*, N], [*] -> [*]
    def forward(self, logits: Tensor, tags: Union[int, List[int]]):
        # negative log likelihood
        arr_tags = np.asarray(tags)  # [*]
        arr_logprobs = log_softmax(logits.data)  # [*, N]
        if len(arr_logprobs.shape) == 1:
            arr_nll = - arr_logprobs[arr_tags]  # []
        else:
            assert len(arr_logprobs.shape) == 2
            arr_nll = - arr_logprobs[np.arange(len(arr_logprobs.shape[0])), arr_tags]  # [*]
        loss_t = Tensor(arr_nll)
        self.store_ctx(logits=logits, loss_t=loss_t, arr_tags=arr_tags, arr_logprobs=arr_logprobs)
        return loss_t

    def backward(self):
        logits, loss_t, arr_tags, arr_logprobs = self.get_ctx('logits', 'loss_t', 'arr_tags', 'arr_logprobs')
        if loss_t.grad is not None:
            arr_probs = np.exp(arr_logprobs)  # [*, N]
            grad_logits = arr_probs  # prob-1 for gold, prob for non-gold
            if len(grad_logits.shape) == 1:
                grad_logits[arr_tags] -= 1.
                grad_logits *= loss_t.grad
            else:
                grad_logits[np.arange(len(grad_logits.shape[0])), arr_tags] -= 1.
                grad_logits *= loss_t.grad[:,None]
            logits.accumulate_grad(grad_logits)
        # --

class OpAdd(Op):
    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, b: Tensor, alpha_a=1., alpha_b=1.):
        if b is None:
            arr_add = alpha_a * a.data
        else:
            arr_add = alpha_a * a.data + alpha_b * b.data
        t_add = Tensor(arr_add)
        self.store_ctx(a=a, b=b, t_add=t_add, alpha_a=alpha_a, alpha_b=alpha_b)
        return t_add

    def backward(self):
        a, b, t_add, alpha_a, alpha_b = self.get_ctx('a', 'b', 't_add', 'alpha_a', 'alpha_b')
        if t_add.grad is not None:
            a.accumulate_grad(alpha_a * t_add.grad)
            if b is not None:
                b.accumulate_grad(alpha_b * t_add.grad)
        # --

class OpLookup(Op):
    def __init__(self):
        ### Add your implementation below and comment the following line.
        # raise NotImplementedError
        super().__init__()

    def forward(self, t: Tensor, ind):
        if not isinstance(ind, np.ndarray):
            ind = np.asarray(ind)
        t_lookup = Tensor(t.data[ind])
        self.store_ctx(t=t, t_lookup=t_lookup, ind=ind)
        return t_lookup

    def backward(self):
        t, t_lookup, ind = self.get_ctx('t', 't_lookup', 'ind')
        if t_lookup.grad is not None:
            grad = [(int(i), g) for i, g in zip(ind, t_lookup.grad)]
            t.accumulate_grad_sparse(grad)

class OpDot(Op):
    def __init__(self):
        ### Add your implementation below and comment the following line.
        # raise NotImplementedError
        super().__init__()

    def forward(self, t1: Tensor, t2: Tensor):
        dot = np.dot(t1.data, t2.data)
        dot_t = Tensor(dot)
        self.store_ctx(t1=t1, t2=t2, dot_t = dot_t, t1_d = t1.data.copy(), t2_d = t2.data.copy())
        return dot_t

    def backward(self):
        t1, t2, dot_t, t1_d, t2_d = self.get_ctx('t1', 't2', 'dot_t', 't1_d', 't2_d')
        if dot_t.grad is not None:
            t1_grad = np.outer(dot_t.grad, t2_d) # dot_t.grad.T.dot(t2_d)
            # t1_grad = dot_t.grad.T.dot(t2_d)
            t2_grad = t1_d.T.dot(dot_t.grad)     # np.outer(t1_d, dot_t.grad)
            # t2_grad = np.outer(t1_d, dot_t.grad)
            t1.accumulate_grad(t1_grad)
            t2.accumulate_grad(t2_grad)

class OpTanh(Op):
    def __init__(self):
        ### Add your implementation below and comment the following line.
        # raise NotImplementedError
        super().__init__()

    def forward(self, t: Tensor):
        arr_tanh = np.tanh(t.data)
        t_tanh = Tensor(arr_tanh)
        self.store_ctx(t=t, arr_tanh=arr_tanh, t_tanh=t_tanh)
        return t_tanh


    def backward(self):
        t, arr_tanh, t_tanh = self.get_ctx('t', 'arr_tanh', 't_tanh')
        if t_tanh.grad is not None:
            # t.accumulate_grad()
            grad_t = (1 - arr_tanh**2) * t_tanh.grad
            t.accumulate_grad(grad_t)

class OpMax(Op):
    def __init__(self):
        ### Add your implementation below and comment the following line.
        # raise NotImplementedError
        super().__init__()

    def forward(self, t: Tensor, axis):
        max_id = np.argmax(t.data, axis)
        t_max = Tensor(np.max(t.data, axis))
        self.store_ctx(t=t, t_max=t_max, max_id=max_id, axis=axis)
        return t_max

    def backward(self):
        t, t_max, max_id, axis = self.get_ctx('t', 't_max', 'max_id', 'axis')
        if t_max.grad is not None:
            ind = np.reshape(max_id, -1)
            g = np.zeros_like(t_max.grad, shape=(ind.shape[0], t.data.shape[axis]))
            g[np.arange(ind.shape[0]), ind] = t_max.grad
            g = np.reshape(g, (t_max.grad.shape[0], -1))
            g = np.swapaxes(g, axis, -1)
            t.accumulate_grad(g)

class OpAvg(Op):
    # NOTE: Implementation of OpAvg is optional, it can be skipped if you wish
    def __init__(self):
        ### Add your implementation below and comment the following line.
        # raise NotImplementedError
        super().__init__()

    # def forward(self, t: Tensor):
        # pass

    # def backward(self):
        # pass

class ComputationGraph:
    """
    This class is the one that keeps track of the current computational graph.

    It simply contains a list of Ops, which are registered in Op.__init__()

    In forward, these Op are appended incrementally in calculation order, and in backward (see function backward, they are visited in reversed order).

    """

    # global cg
    _cg: 'ComputationGraph' = None

    @classmethod
    def get_cg(cls, reset=False):
        if ComputationGraph._cg is None or reset:
            ComputationGraph._cg = ComputationGraph()
        return ComputationGraph._cg

    def __init__(self):
        self.ops: List[Op] = []  # list of ops by execution order

    def reg_op(self, op: Op):
        assert op.idx is None
        op.idx = len(self.ops)
        self.ops.append(op)


class Initializer:
    """
    This is simply a collection of initializer methods that produces a np.ndarray according to the specified shape and other parameters like initializer ranges.

    """

    @staticmethod
    def uniform(shape: Sequence[int], a=0.0, b=0.2):
        return np.random.uniform(a, b, size=shape)

    @staticmethod
    def normal(shape: Sequence[int], mean=0., std=0.02):
        return np.random.normal(mean, std, size=shape)

    @staticmethod
    def constant(shape: Sequence[int], val=0.):
        return np.full(shape, val)

    @staticmethod
    def xavier_uniform(shape: Sequence[int], gain=1.0):
        """
        This accepts inputs of shape and gain, and outputs a np.ndarray where the shape is shape.
        gain simply means that finally we are scaling the weights by this value.

        See Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010) for details about Xavier/Glorot initialization, 
        and this blog for more details about initialization in general: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
 
        """

        ### Add your implementation below and comment the following line.
        # raise NotImplementedError
        xav = 6/(shape[0]+shape[1])
        scale = gain* np.sqrt(xav)
        return np.random.uniform(-scale, scale, size=shape)


class Model:
    """
    Model maintains a collection of Parameter. We provide Model.add_parameters() as a shortcut of making a new Parameter and adding it to the model.

    """

    def __init__(self):
        self.params: List[Parameter] = []

    def add_parameters(self, shape, initializer='normal', **initializer_kwargs):
        init_f = getattr(Initializer, initializer)
        data = init_f(shape, **initializer_kwargs)
        param = Parameter(data)
        self.params.append(param)
        return param

    def save(self, path: str):
        data = {f"p{i}": p.data for i,p in enumerate(self.params)}
        np.savez(path, **data)

    def load(self, path: str):
        data0 = np.load(path)
        data = {int(n[1:]):d for n,d in data0.items()}
        for i,p in enumerate(self.params):
            d = data[i]
            assert d.shape == p.shape
            p.data = d


class Trainer:
    """
    Trainer takes a Model and handles the update of the parameters. Trainer.update() denotes one update step which will be implemented in the sub-classes.

    """

    def __init__(self, model: Model):
        self.model = model

    def clone_param_stats(self, model: Model):
        clone = list()
        for param in model.params:
            clone.append(np.zeros(param.data.shape))
        return clone

    def update(self):
        # you will override this function in a subclass below
        raise NotImplementedError()


class SGDTrainer(Trainer):
    """
    SGDTrainer is a simple SGD trainer, notice that here we check whether Tensor.grad is sparse (simulated by a python dictionary) or not, and update accordingly. (In our enviroment with CPU, enabling sparse update is much faster, but not necessarily with GPU).

    Notice that at the end of each update, we also clear the gradients (clearing by simply setting Tensor.grad=None). This can usually be two separate steps, but we combine them here for convenience.

    """

    def __init__(self, model: Model, lrate=0.1):
        super().__init__(model)
        self.lrate = lrate

    def update(self):
        lrate = self.lrate
        for p in self.model.params:
            if p.grad is not None:
                if isinstance(p.grad, dict):  # sparsely update to save time!
                    self.update_sparse(p, p.grad, lrate)
                else:
                    self.update_dense(p, p.grad, lrate)
            # clean grad
            p.grad = None

    def update_dense(self, p: Parameter, g: np.ndarray, lrate: float):
        p.data -= lrate * g

    def update_sparse(self, p: Parameter, gs: Dict[int, np.ndarray], lrate: float):
        for widx, arr in gs.items():
            p.data[widx] -= lrate * arr


class MomentumTrainer(Trainer):
    """
    Notice that in this one, there can be some variations. You can implement according to this formula:
        m <- mrate*m + (1-mrate)*g, p <- p - lrate * m
    but if you find something better feel free to use that as well.
    
    Notice that for update_sparse, we still need to update the parameters if there are historical m, even if there are no gradients for the current step.
    
    Please remember to clear gradients (by setting p.grad=None) at the end of update, similar to SGDTrainer.

    """

    def __init__(self, model: Model, lrate=0.1, mrate=0.99):
        ### Add your implementation below and comment the following line.
        # raise NotImplementedError
        super().__init__(model)
        self.lrate = lrate
        self.mrate = mrate
        self.momentum = []

    def update(self):
        lr = self.lrate
        mr = self.mrate

        if not len(self.momentum):
            self.momentum = [np.zeros_like(each.data) for each in self.model.params]

        for i, each in enumerate(self.model.params):
            if each.grad is None:
                continue
            if isinstance(each.grad, dict):
                self.update_sparse(each, i, each.grad, lr, mr)
            else:
                self.update_dense(each, i, each.grad, lr, mr)
            each.grad = None

    def update_dense(self, p, id, g, lr, mr):
        tmp = mr * self.momentum[id] + (1-mr) *g
        p.data -= lr * tmp
        self.momentum[id] = tmp

    def update_sparse(self, p, id, g, lr, mr):
        curr = self.momentum[id] * mr
        for widx, arr in g.items():
            curr[widx] += (1-mr) * arr
        p.data -= lr * curr
        self.momentum[id] = curr


### Graph computation functions

def reset_computation_graph():
    """
    reset_computation_graph discards the previous ComputationGraph (together with previous Ops and intermediate Tensors) and make a new one. This should be called at the start of each computation loop.
    """
    return ComputationGraph.get_cg(reset=True)

def forward(t: Tensor):
    """
    forward gets the np.ndarray value of a Tensor. Since we calculate everything greedily, this step is simply retriving the Tensor.data.
    """
    return np.asarray(t.data)

def backward(t: Tensor, alpha=1.):
    """
    backward assign a scalar gradient alpha to a tensor and do backwards according to the reversed order of the Op list stored inside ComputationGraph.
    """
    # first put grad to the start one
    t.accumulate_grad(alpha)
    # locate the op
    op = t.op
    assert op is not None, "Cannot backward on tensor since no op!!"
    # backward the whole graph!!
    cg = ComputationGraph.get_cg()
    for idx in reversed(range(op.idx+1)):
        cg.ops[idx].backward()


### Helper
def log_softmax(x: np.ndarray, axis=-1):
    c = np.max(x, axis=axis, keepdims=True)  # [*, 1, *]
    x2 = x - c  # [*, ?, *]
    logsumexp = np.log(np.exp(x2).sum(axis=axis, keepdims=True))  # [*, 1, *]
    return x2 - logsumexp

### Finally, here are some shortcut functions to make it more convenient.
def lookup(W_emb, words): return OpLookup().full_forward(W_emb, words)
def sum(emb, axis): return OpSum().full_forward(emb, axis)
def dot(W_h_i, h): return OpDot().full_forward(W_h_i, h)
def tanh(param): return OpTanh().full_forward(param)
def relu(param): return OpRelu().full_forward(param)
def log_loss(my_scores, tag): return OpLogloss().full_forward(my_scores, tag)
def dropout(x, drop, is_training): return OpDropout().full_forward(x, drop, is_training)
def avg(x, axis): return OpAvg().full_forward(x, axis)
def max(x, axis): return OpMax().full_forward(x, axis)
