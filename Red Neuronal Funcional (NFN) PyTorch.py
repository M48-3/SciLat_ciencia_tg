"""
En este ejemplo, se define la clase NFN que hereda de la clase torch.autograd.Function. Esta clase define las funciones forward y backward que se utilizan para calcular los resultados de la red y los gradientes respectivamente. La función forward calcula la multiplicación matricial entre la entrada y los pesos, mientras que la función backward calcula los gradientes de la entrada y los pesos.
Para utilizar esta clase, se crea un tensor de entrada x y un tensor de pesos w. Luego, se llama a la función NFN.apply() para calcular la salida de la red y. Finalmente, se calculan los gradientes de y con respecto a x y w utilizando la función y.sum().backward().
"""

import torch

class NFN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        return grad_input, grad_weight

x = torch.randn(3, 5, requires_grad=True)
w = torch.randn(2, 5, requires_grad=True)
y = NFN.apply(x, w)
y.sum().backward()

print(x.grad)
print(w.grad)
