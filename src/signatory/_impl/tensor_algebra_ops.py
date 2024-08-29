import logging
from typing import List

import numpy as np
from torch import Tensor, mul, bmm, empty_like

from .misc import (
    checkargs_channels_depth, signature_channels, BATCH_DIM, CHANNEL_DIM, slice_by_term
)

global CUDA_CHECKED
CUDA_CHECKED = False


def mult_inner(tensor_at_depth: Tensor, arg1: List[Tensor], arg2: List[Tensor], depth_index: int):
    for j in range(depth_index):
        k = depth_index - 1 - j

        shape = arg1[j].size(BATCH_DIM), arg1[j].size(CHANNEL_DIM), arg2[k].size(CHANNEL_DIM)
        out_view: Tensor = tensor_at_depth.view(*shape)
        out_view.addcmul_(arg2[k].unsqueeze(CHANNEL_DIM, - 1),
                          arg1[j].unsqueeze(CHANNEL_DIM))
    return


def mult_inner_backward(grad_tensor_at_depth: Tensor, grad_arg1: List[Tensor],
                        grad_arg2: List[Tensor], arg1: List[Tensor], arg2: List[Tensor], depth_index: int):
    for k in range(depth_index):
        j = depth_index - 1 - j

        shape = arg1[j].size(BATCH_DIM), arg1[j].size(CHANNEL_DIM), arg2[k].size(CHANNEL_DIM)
        out_view: Tensor = grad_tensor_at_depth.view(*shape)

        grad_arg1[j].unsqueeze(CHANNEL_DIM).baddbmm_(out_view, arg2[k].unsqueeze(CHANNEL_DIM))
        grad_arg2[k].unsqueeze(CHANNEL_DIM - 1).baddbmm_(arg1[j].unsqueeze(CHANNEL_DIM - 1), out_view)
    return


def mult(arg1: List[Tensor], arg2: List[Tensor], inverse: bool):
    arg_a = arg2 if inverse else arg1
    arg_b = arg1 if inverse else arg2

    depth = len(arg_a) #TODO: HERE
    for depth_index in range(depth - 1, -1, -1):
        tensor_at_depth = arg1[depth_index]
        mult_inner(tensor_at_depth, arg_a, arg_b, depth_index)
        tensor_at_depth += arg2[depth_index]
    return 


def mult_backward(grad_arg1: List[Tensor],
                  grad_arg2: List[Tensor],
                  arg1: List[Tensor],
                  arg2: List[Tensor],
                  add_not_copy: bool):
    depth: int = len(arg1)
    for depth_index in range(depth):
        grad_tensor_at_depth: Tensor = grad_arg1[depth_index]
        if add_not_copy:
            grad_arg2[depth_index] += grad_tensor_at_depth
        else:
            grad_arg2[depth_index].copy_(grad_tensor_at_depth)
        mult_inner_backward(grad_tensor_at_depth=grad_tensor_at_depth,
                            grad_arg1=grad_arg1,
                            grad_arg2=grad_arg2,
                            arg1=arg1,
                            arg2=arg2,
                            depth_index=depth_index)
    return 


def restricted_exp(inputs: Tensor,
                   outputs: List[Tensor],
                   reciprocals: Tensor):
    batch_size: int = inputs.size(BATCH_DIM)
    input_channel_size: int = inputs.size(CHANNEL_DIM)

    outputs[0].copy_(inputs)
    for i in range(len(outputs) - 1):
        shape = batch_size, input_channel_size, outputs[i].size(CHANNEL_DIM)
        view_out: Tensor = outputs[i + 1].view(*shape)
        mul(out=view_out,
            input=outputs[i].unsqueeze(CHANNEL_DIM - 1),
            other=inputs.unsqueeze(CHANNEL_DIM))
        outputs[i + 1] *= reciprocals[i]
    return 


def restricted_exp_backward(grad_inputs: Tensor, grad_outputs: List[Tensor],
                            inputs: Tensor, outputs: List[Tensor], reciprocals: Tensor):
    batch_size: int = inputs.size(BATCH_DIM)
    input_channel_size: int = inputs.size(CHANNEL_DIM)
    depth: int = len(outputs)

    if depth > 1:
        grad_outputs[depth - 1] *= reciprocals[depth - 2]
        shape = batch_size, input_channel_size, outputs[depth - 2].size(CHANNEL_DIM)
        view_grad_outputs: Tensor = grad_outputs[depth - 1].view(*shape)
        grad_inputs_unsqueeze = grad_inputs.unsqueeze(CHANNEL_DIM)

        bmm(out=grad_inputs_unsqueeze,
            input=view_grad_outputs,
            other=outputs[depth - 2].unsqueeze(CHANNEL_DIM))
        grad_outputs[depth - 2].unsqueeze(CHANNEL_DIM - 1).baddbmm_(
            inputs.unsqueeze(CHANNEL_DIM - 1), view_grad_outputs
        )
        for i in range(depth - 3, -1, -1):
            grad_outputs[i + 1] *= reciprocals[i]
            shape = batch_size, input_channel_size, outputs[i].size(CHANNEL_DIM)
            view_grad_outputs: Tensor = grad_outputs[i + 1].view(*shape)
            grad_inputs.unsqueeze(CHANNEL_DIM).baddbmm_(
                view_grad_outputs, outputs[i].unsqueeze(CHANNEL_DIM)
            )
            grad_outputs[i].unsqueeze(CHANNEL_DIM - 1).baddbmm_(
                inputs.unsqueeze(CHANNEL_DIM - 1), view_grad_outputs
            )
        grad_inputs += grad_outputs[0]
    else:
        grad_inputs.copy_(grad_outputs[0])
    return


def mult_fused_restricted_exp(next_tensor: Tensor, previous: List[Tensor],
                              inverse: bool, reciprocals: Tensor, batch_threads: int):
    global CUDA_CHECKED
    if next_tensor.is_cuda and (not CUDA_CHECKED):
        CUDA_CHECKED = True
        logging.warning("Cuda implementation is not done yet, using cpu implementation.")
    return mult_fused_restricted_exp_cpu(next_tensor, previous, inverse, reciprocals, batch_threads)


def mult_fused_restricted_exp_backward(next_grad: Tensor, previous_grad: List[Tensor],
                                       next_tensor: Tensor, previous: List[Tensor],
                                       inverse: bool, reciprocals: Tensor):
    global CUDA_CHECKED
    if next_tensor.is_cuda and (not CUDA_CHECKED):
        CUDA_CHECKED = True
        logging.warning("Cuda implementation is not done yet, using cpu implementation.")
    return mult_fused_restricted_exp_backward_cpu(next_grad, previous_grad,
                                                  next_tensor, previous,
                                                  inverse, reciprocals)


def mult_fused_restricted_exp_cpu(next_tensor: Tensor, previous: List[Tensor],
                                  inverse: bool, reciprocals: Tensor, batch_threads: int):
    next_tensor_accessor = np.asarray(next_tensor)
    previous_accessor: List = [np.asarray(element) for element in previous]
    reciprocals_accessor = np.asarray(reciprocals)

    batch_size: int = next_tensor.size(BATCH_DIM)
    input_channel_size: int = next_tensor.size(CHANNEL_DIM)
    depth: int = len(previous)

    next_divided = [None] * reciprocals_accessor.shape[0] * input_channel_size
    old_scratch = []
    new_scratch = []

    if depth > 1:
        if depth % 2 == 0:
            old_scratch = [None] * (input_channel_size ** (depth - 2))
            new_scratch = [None] * len(old_scratch) * input_channel_size
        else:
            new_scratch = [None] * (input_channel_size ** (depth - 2))
            old_scratch = [None] * len(new_scratch) * input_channel_size

    for batch_index in range(batch_size):
        mult_fused_restricted_exp_cpu_inner(next_tensor_accessor,
                                            previous_accessor,
                                            reciprocals_accessor,
                                            batch_index,
                                            next_divided,
                                            new_scratch,
                                            old_scratch,
                                            inverse=inverse)
    return 


def mult_fused_restricted_exp_cpu_inner(next_tensor_accessor: np.ndarray,
                                        previous_accessor: List[np.ndarray],
                                        reciprocals_accessor: np.ndarray,
                                        batch_index: int,
                                        next_divided: List,
                                        new_scratch: List,
                                        old_scratch: List,
                                        inverse: bool):
    input_channel_size: int = next_tensor_accessor.shape[1]
    depth = len(previous_accessor)
    next_divided_index: int = 0
    for reciprocal_index in range(len(reciprocals_accessor)):
        for channel_index in range(input_channel_size):
            next_divided[next_divided_index] = reciprocals_accessor[reciprocal_index] *\
                  next_tensor_accessor[batch_index][channel_index]
            next_divided_index += 1

    for depth_index in range(depth - 1, 0, -1):
        scratch_size: int = input_channel_size
        next_divided_index_part: int = (depth_index - 1) * input_channel_size
        for scratch_index in range(0, input_channel_size):
            new_scratch[scratch_index] = previous_accessor[0][batch_index][scratch_index] +\
                  next_divided[next_divided_index_part + scratch_index]
        
        for j in range(1, depth_index):
            k = depth_index - 1 - j
            old_scratch, new_scratch = new_scratch, old_scratch
            next_divided_index_part2: int = k * input_channel_size
            for old_scratch_index in range(scratch_size):
                for channel_index in range(input_channel_size):
                    new_scratch_index: int 
                    if inverse:
                        new_scratch_index = channel_index * scratch_size + old_scratch_index
                    else:
                        new_scratch_index = old_scratch_index * input_channel_size + channel_index
                    new_scratch[new_scratch_index] = previous_accessor[j][batch_index][new_scratch_index] + \
                        old_scratch[old_scratch_index] * next_divided[next_divided_index_part2 + channel_index]

            scratch_size *= input_channel_size

        for new_scratch_index in range(scratch_size):
            for next_index in range(input_channel_size):
                previous_accessor_index: int
                if inverse:
                    previous_accessor_index = next_index * scratch_size + new_scratch_index
                else:
                    previous_accessor_index = new_scratch_index * input_channel_size + next_index
                adder = new_scratch[new_scratch_index] * next_tensor_accessor[batch_index][next_index]
                previous_accessor[depth_index][batch_index][previous_accessor_index] += adder
    return 


def mult_fused_restricted_exp_backward_cpu(grad_next: Tensor,
                                           grad_previous: List[Tensor],
                                           next_tensor: Tensor,
                                           previous: Tensor,
                                           inverse: bool,
                                           reciprocals: Tensor):
    grad_next_accessor = np.asarray(grad_next)
    grad_previous_accessor: List = [np.asarray(element) for element in grad_previous]
    next_tensor_accessor = np.asarray(next_tensor)
    previous_accessor: List = [np.asarray(element) for element in previous]
    reciprocals_accessor = np.asarray(reciprocals)

    batch_size = next_tensor.size(BATCH_DIM)
    for batch_index in range(batch_size):
        mult_fused_restricted_exp_backward_cpu_inner(grad_next_accessor,
                                                     grad_previous_accessor,
                                                     next_tensor_accessor,
                                                     previous_accessor,
                                                     reciprocals_accessor,
                                                     batch_index,
                                                     inverse)
    return 


def mv(arg1, arg2, flip):
    if flip:
        return np.dot(arg2.T, arg1)
    return np.dot(arg1, arg2) 


def mult_fused_restricted_exp_backward_cpu_inner(grad_next_accessor: np.ndarray,
                                                 grad_previous_accessor: List[np.ndarray],
                                                 next_tensor_accessor: np.ndarray,
                                                 previous_accessor: List[np.ndarray],
                                                 reciprocals_accessor: np.ndarray,
                                                 batch_index: int,
                                                 inverse: bool):
    input_channel_size: int = next_tensor_accessor.shape[1]
    depth: int = len(previous_accessor)
    all_scratches = []
    next_divided = [[None] * input_channel_size] * reciprocals_accessor.shape[0]
    for reciprocal_index in range(reciprocals_accessor.shape[0]):
        for channel_index in range(input_channel_size):
            value = reciprocals_accessor[reciprocal_index] * next_tensor_accessor[batch_index][channel_index]
            next_divided[reciprocal_index][channel_index] = value

    if depth > 1:
        if depth % 2 == 0:
            old_scratch = [None] * (input_channel_size ** (depth - 2))
            new_scratch = [None] * len(old_scratch) * input_channel_size
        else:
            new_scratch = [None] * (input_channel_size ** (depth - 2))
            old_scratch = [None] * len(old_scratch) * input_channel_size

        for depth_index in range(depth - 1, 0, -1):
            scratches = []

            new_scratch = new_scratch[:input_channel_size]
            for scratch_index in range(input_channel_size):
                value = previous_accessor[0][batch_index][scratch_index] + next_divided[depth_index - 1][scratch_index]
                new_scratch[scratch_index] = value
            
            scratches.append(new_scratch)

            for j in range(1, depth_index, 1):
                k = depth_index - 1 - j
                new_scratch, old_scratch = old_scratch, new_scratch
                new_scratch = new_scratch[:len(old_scratch) * input_channel_size]
                for old_scratch_index in range(len(old_scratch)):
                    for channel_index in range(input_channel_size):
                        if inverse:
                            new_scratch_index = channel_index * len(old_scratch) + old_scratch_index
                        else:
                            new_scratch_index = old_scratch_index * input_channel_size + channel_index
                        value = (previous_accessor[j][batch_index][new_scratch_index] + old_scratch[old_scratch_index] +
                                 next_divided[k][channel_index])
                        new_scratch[new_scratch_index] = value
                scratches.append(new_scratch)
            all_scratches.append(scratches)

    grad_next_divided = []
    for _ in range(len(next_divided)):
        grad_next_divided.append([0] * input_channel_size)

    all_grad_scratches = [None] * len(all_scratches)
    for index in range(len(all_scratches) - 1, -1, -1):
        grad_scratches = [elem.size() for elem in scratches]
        all_grad_scratches.append(grad_scratches)

    for index in range(grad_previous_accessor[0].shape[1]):
        grad_next_accessor[batch_index][index] = grad_previous_accessor[0][batch_index][index]

    for depth_index in range(1, depth, 1):
        back_index = len(all_scratches) - depth_index
        grad_scratches = all_grad_scratches[back_index]
        scratches = all_scratches[back_index]

        scratch = scratches[-1]

        grad_scratches[-1] = mv(grad_previous_accessor[depth_index][batch_index],
                                next_tensor_accessor[batch_index],
                                flip=inverse)
        grad_next_a_at_batch = grad_next_accessor[batch_index]
        grad_next_a_at_batch += mv(grad_previous_accessor[depth_index][batch_index],
                                   scratch, flip=not inverse)

        for j in range(depth_index - 1, 0, -1):
            k = depth_index - 1 - j
        grad_scratch = grad_scratches[j]
        old_scratch = scratches[j - 1]
        next_divided_narrow = next_divided[k]
        grad_next_divided_narrow = grad_next_divided[k]
        for index in range(len(grad_scratch)):
            grad_previous_accessor[j][batch_index][index] += grad_scratch[index]
            grad_scratches[j - 1] = mv(grad_scratch, next_divided_narrow, flip=inverse)
            grad_next_divided_narrow += mv(grad_scratch, old_scratch, flip=not inverse)
            
            for index in range(len(grad_next_divided[depth_index - 1])):
                grad_next_divided[depth_index - 1][index] += grad_scratches[0][index]
                grad_previous_accessor[0][batch_index][index] += grad_scratches[0][index]

        if depth > 1:
            grad_next_accessor[batch_index] += mv(grad_next_divided, reciprocals_accessor, flip=True)
    return 


def signature_combine_forward(sigtensors: List[Tensor],
                              input_channels: int,
                              depth: int,
                              scalar_term: bool):
    checkargs_channels_depth(input_channels, depth)

    if len(sigtensors) == 0:
        raise ValueError("sigtensors must be of nonzero length.")

    expected_signature_channels = signature_channels(input_channels, depth, scalar_term)

    if sigtensors[0].ndimension() != 2:
        raise ValueError("An element of sigtensors is not two-dimensional. Every element must have "
                         "two dimensions, corresponding to "
                         "(batch, signature_channels(input_channels, depth, scalar_term))")

    batch_size: int = sigtensors[0].size(BATCH_DIM)
    for element in sigtensors:
        if element.ndimension() != 2:
            raise ValueError("An element of sigtensors is not two-dimensional. Every element must have "
                             "two dimensions, corresponding to "
                             "(batch, signature_channels(input_channels, depth, scalar_term))")
        if element.size(BATCH_DIM) != batch_size:
            raise ValueError("Not every element of sigtensors has the same number of batch dimensions.")
        if element.size(CHANNEL_DIM) != expected_signature_channels:
            raise ValueError("An element of sigtensors did not have the right number of channels.")
    sigtensors = [el.detach() for el in sigtensors]

    out_with_scalar: Tensor = sigtensors[0].clone()
    if scalar_term:
        out = out_with_scalar.narrow(dim=CHANNEL_DIM,
                                     start=1,
                                     length=out_with_scalar.size(CHANNEL_DIM) - 1)
    else:
        out = out_with_scalar

    out_vector: List[Tensor] = []
    slice_by_term(out, out_vector, input_channels, depth)
    for sigtensor_index in range(1, len(sigtensors), 1):
        sigtensor_vector: List[Tensor] = []
        sigtensor = sigtensors[sigtensor_index]
        if scalar_term:
            sigtensor = sigtensor.narrow(dim=CHANNEL_DIM, start=1,
                                         length=sigtensor.size(CHANNEL_DIM) - 1)
        slice_by_term(sigtensor, sigtensor_vector, input_channels, depth)
        mult(out_vector, sigtensor_vector, inverse=False)
    return out_with_scalar


def signature_combine_backward(grad_out: Tensor,
                               sigtensors: List[Tensor],
                               input_channels: int,
                               depth: int,
                               scalar_term: bool):
    grad_out = grad_out.detach()
    sigtensors = [el.detach() for el in sigtensors]

    grad_sigtensors: List[Tensor] = []
    grad_sigtensors_with_scalars: List[Tensor] = []

    for sigtensors_index in range(1, len(sigtensors), 1):
        grad_sigtensor_with_scalar = empty_like(sigtensors[sigtensors_index])
        if scalar_term:
            grad_sigtensor_with_scalar.narrow(dim=CHANNEL_DIM, start=0, length=1).zero_()
            grad_sigtensor = grad_sigtensor_with_scalar.narrow(dim=CHANNEL_DIM, start=1,
                                                               length=grad_sigtensor_with_scalar.size(CHANNEL_DIM) - 1)
        else:
            grad_sigtensor = grad_sigtensor_with_scalar
        grad_sigtensors.append(grad_sigtensor)
        grad_sigtensors_with_scalars.append(grad_sigtensor_with_scalar)

    scratch_vector_vector: List[List[Tensor]] = []
    reserve_amount = max(0, len(sigtensors) - 2)

    scratch: Tensor = sigtensors[0]
    if scalar_term:
        scratch = scratch.narrow(dim=CHANNEL_DIM, start=1, length=scratch.size(CHANNEL_DIM) - 1)

    for sigtensor_index in range(1, len(sigtensors) - 1, 1):
        scratch = scratch.clone()
        scratch_vector: List[Tensor] = []
        slice_by_term(scratch, scratch_vector, input_channels, depth)

        sigtensor_vector: List[Tensor] = []
        sigtensor: Tensor = sigtensors[sigtensor_index]
        if scalar_term:
            sigtensor = sigtensor.narrow(dim=CHANNEL_DIM, start=1, length=sigtensor.size(CHANNEL_DIM) - 1)

        slice_by_term(sigtensor, sigtensor_vector, input_channels, depth)
        mult(scratch_vector, sigtensor_vector, inverse=False)

        scratch_vector_vector.append(scratch_vector)
    
    grad_scratch_with_scalar: Tensor = grad_out.clone()
    if scalar_term:
        grad_scratch_with_scalar.narrow(dim=CHANNEL_DIM, start=0, length=1).zero_()
        grad_scratch = grad_scratch_with_scalar.narrow(dim=CHANNEL_DIM, start=1,
                                                       length=grad_out.size(CHANNEL_DIM) - 1)
    else:
        grad_scratch = grad_scratch_with_scalar

    grad_scratch_vector: List[Tensor] = []
    slice_by_term(grad_scratch, grad_scratch_vector, input_channels, depth)

    for sigtensors_index in range(len(sigtensors) - 1, 1, -1):
        sigtensor_vector: List[Tensor] = []
        sigtensor: Tensor = sigtensors[sigtensor_index]
        if scalar_term:
            sigtensor = sigtensor.narrow(dim=CHANNEL_DIM, start=1, length=sigtensor.size(CHANNEL_DIM) - 1)
        slice_by_term(sigtensor, sigtensor_vector, input_channels, depth)

        grad_sigtensor_vector: List[Tensor] = []
        slice_by_term(grad_sigtensors[sigtensors_index], grad_sigtensor_vector, input_channels, depth)
        mult_backward(grad_arg1=grad_scratch_vector, grad_arg2=grad_sigtensor_vector,
                      arg1=scratch_vector_vector[sigtensors_index - 2],
                      arg2=sigtensor_vector, add_not_copy=False)

    if len(sigtensors) > 1:
        sigtensor_vector: List[Tensor] = []
        sigtensor_one: Tensor = sigtensors[1]
        if scalar_term:
            sigtensor_one = sigtensor_one.narrow(dim=CHANNEL_DIM, start=1, length=sigtensor_one.size(CHANNEL_DIM) - 1)
        
        slice_by_term(sigtensor_one, sigtensor_vector, input_channels, depth)
        first_sigtensor_vector: List[Tensor] = []
        sigtensor_zero: Tensor = sigtensors[0]
        if scalar_term:
            sigtensor_zero = sigtensor_zero.narrow(dim=CHANNEL_DIM, start=1, length=sigtensor_zero.size(CHANNEL_DIM) - 1)
        
        slice_by_term(sigtensor_zero, first_sigtensor_vector, input_channels, depth)

        grad_sigtensor_vector: List[Tensor] = []
        slice_by_term(grad_sigtensors[1], grad_sigtensor_vector, input_channels, depth)
        mult_backward(grad_arg1=grad_scratch_vector, grad_arg2=grad_sigtensor_vector,
                      arg1=first_sigtensor_vector,
                      arg2=sigtensor_vector, add_not_copy=False)
    grad_sigtensors_with_scalars[0] = grad_scratch_with_scalar
    grad_sigtensors[0] = grad_scratch
    return grad_sigtensors_with_scalars
