from typing import List

from torch import Tensor, empty_like, empty

from .misc import (
    STREAM_DIM, BATCH_DIM, CHANNEL_DIM, slice_at_stream, signature_channels, make_reciprocals,
    slice_by_term
)
from .tensor_algebra_ops import (
    mult_fused_restricted_exp, restricted_exp, mult_fused_restricted_exp_backward,
    restricted_exp_backward
)


def compute_path_increments(path: Tensor, basepoint: bool, basepoint_value: Tensor, inverse: bool):
    num_increments: int = path.size(STREAM_DIM) - 1  
    if basepoint:
        if inverse:
            path_increments = empty_like(path)
            path_increments[0].copy_(basepoint_value)
            path_increments.narrow(dim=STREAM_DIM, start=1, length=num_increments).copy_(
                path.narrow(dim=STREAM_DIM, start=0, length=num_increments)
            )
            path_increments -= path
            return path_increments
        else:
            path_increments: Tensor = path.clone()
            path_increments[0] -= basepoint_value
            
            #TODO: improve here, since we change STREAM_DIM = -3, that's first dim
            path_increments[1:num_increments+1] -= path.narrow(dim=STREAM_DIM, start=0, length=num_increments)

            return path_increments
    else:
        if inverse:
            return path.narrow(dim=STREAM_DIM, start=0, length=num_increments) - \
                   path.narrow(dim=STREAM_DIM, start=1, length=num_increments)
        else:
            return path.narrow(dim=STREAM_DIM, start=1, length=num_increments) - \
                   path.narrow(dim=STREAM_DIM, start=0, length=num_increments)


def compute_path_increments_backward(grad_path_increments: Tensor,
                                     basepoint: bool, 
                                     inverse: bool):
    batch_size: int = grad_path_increments.size(BATCH_DIM)
    input_stream_size: int = grad_path_increments.size(STREAM_DIM)
    input_channel_size: int = grad_path_increments.size(CHANNEL_DIM)
    
    if not basepoint:
        input_stream_size += 1
    
    num_increments = input_stream_size - 1
    if basepoint:
        if inverse:
            grad_path = empty_like(grad_path_increments)
            grad_path.narrow(dim=STREAM_DIM, start=0, length=num_increments).copy_(
                grad_path_increments.narrow(dim=STREAM_DIM, start=1, length=num_increments)
            )
            grad_path[-1].zero_()
            grad_path -= grad_path_increments
            return grad_path, grad_path_increments[0]
        else:
            grad_path: Tensor = grad_path_increments.clone()
            #TODO: improve here, since we change STREAM_DIM = -3, that's first dim
            grad_path[0:num_increments] -= grad_path_increments.narrow(dim=STREAM_DIM, start=1, length=num_increments)
            return grad_path, grad_path_increments[0]
    else:
        if inverse:
            grad_path = empty((input_stream_size, batch_size, input_channel_size))
            grad_path[-1].zero_()
            grad_path.narrow(dim=STREAM_DIM, start=0, length=num_increments).copy_(grad_path_increments)
            #TODO: improve here, since we change STREAM_DIM = -3, that's first dim
            grad_path[1:num_increments+1] -=  grad_path_increments
            return grad_path, empty((0,))
        else:
            grad_path = empty((input_stream_size, batch_size, input_channel_size))
            grad_path[0].zero_()
            grad_path.narrow(dim=STREAM_DIM, start=0, length=num_increments).copy_(grad_path_increments)
            #TODO: improve here, since we change STREAM_DIM = -3, that's first dim
            grad_path[0:num_increments] -= grad_path_increments

            return grad_path, empty((0,))


def signature_forward_inner(path_increments: Tensor,
                            reciprocals: Tensor,
                            signature: Tensor,
                            signature_by_term: List[Tensor],
                            signature_by_term_at_stream: List[Tensor],
                            inverse: bool,
                            stream: bool,
                            start: int,
                            end: int, 
                            batch_threads: int):
    for stream_index in range(start, end, 1):
        if stream:
            signature[stream_index].copy_(signature[stream_index - 1])
            slice_at_stream(signature_by_term, signature_by_term_at_stream, stream_index)
        mult_fused_restricted_exp(path_increments[stream_index],
                                  signature_by_term_at_stream,
                                  inverse=inverse,
                                  reciprocals=reciprocals,
                                  batch_threads=batch_threads)
    return 


def signature_checkargs(path: Tensor,
                        depth: int,
                        basepoint: bool, 
                        basepoint_value: Tensor,
                        initial: bool,
                        initial_value: Tensor,
                        scalar_term: bool):
    if path.ndimension() == 2:
        raise ValueError("Argument 'path' must be a 3-dimensional tensor, with dimensions "
                         "corresponding to (batch, stream, channel) respectively. If you just want "
                         "the signature or logsignature of a single path then wrap it in a single "
                         "batch dimension by replacing e.g. `signature(path, depth)` with "
                         "`signature(path.unsqueeze(0), depth).squeeze(0)`.")
    
    if path.ndimension() != 3:
        raise ValueError("Argument 'path' must be a 3-dimensional tensor, with dimensions "
                         "corresponding to (batch, stream, channel) respectively.")
    if (path.size(BATCH_DIM) == 0) or (path.size(STREAM_DIM) == 0) or (path.size(CHANNEL_DIM) == 0):
        raise ValueError("Argument 'path' cannot have dimensions of size zero.")

    if not basepoint and (path.size(STREAM_DIM) == 1):
        raise ValueError("Argument 'path' must have stream dimension of size at least 2. (Need at "
                         "least this many points to define a path.)")

    if depth < 1:
        raise ValueError("Argument 'depth' must be an integer greater than or equal to one.")
    
    if not path.is_floating_point():
        raise ValueError("Argument 'path' must be of floating point type.")

    if basepoint:
        if basepoint_value.ndimension() != 2:
            raise ValueError("Argument 'basepoint' must be a 2-dimensional tensor, corresponding to "
                             "(batch, channel) respectively.")
        if (basepoint_value.size(CHANNEL_DIM) != path.size(CHANNEL_DIM)) or \
            (basepoint_value.size(BATCH_DIM) != path.size(BATCH_DIM)):
            raise ValueError("Arguments 'basepoint' and 'path' must have dimensions of the same "
                             "size.")
        if path.dtype() != basepoint_value.dtype():
            raise ValueError("Argument 'basepoint' does not have the same dtype as 'path'.")
        
    if initial:
        if initial_value.ndimension() != 2:
            raise ValueError("Argument 'initial' must be a 2-dimensional tensor, corresponding to "
                             "(batch, signature_channels) respectively.")
        if (initial_value.size(CHANNEL_DIM) != signature_channels(path.size(CHANNEL_DIM), depth, scalar_term)) or \
            (initial_value.size(BATCH_DIM) != path.size(BATCH_DIM)):
            raise ValueError("Argument 'initial' must have correctly sized batch and channel "
                             "dimensions.")
        if path.dtype() != initial_value.dtype():
            raise ValueError("Argument 'initial' does not have the same dtype as 'path'.")
    return 


def signature_forward(path: Tensor,
                      depth: int,
                      stream: bool,
                      basepoint: bool,
                      basepoint_value: Tensor,
                      inverse: bool,
                      initial: bool,
                      initial_value: Tensor,
                      scalar_term: bool):
    signature_checkargs(path, depth, basepoint, basepoint_value, initial, initial_value, scalar_term)

    path = path.detach()
    basepoint_value = basepoint_value.detach()
    initial_value = initial_value.detach()

    if scalar_term and initial:
        initial_value = initial_value.narrow(
            dim=CHANNEL_DIM, strt=1, length=initial_value.size(CHANNEL_DIM) - 1
        )

    batch_size: int = path.size(BATCH_DIM)
    input_stream_size: int = path.size(STREAM_DIM)
    input_channel_size: int = path.size(CHANNEL_DIM)
    output_stream_size: int = path.size(STREAM_DIM) - (1 - int(basepoint))
    output_channel_size: int = signature_channels(input_channel_size, depth, False);

    reciprocals = make_reciprocals(depth)
    path_increments: Tensor = compute_path_increments(path=path,
                                                      basepoint=basepoint,
                                                      basepoint_value=basepoint_value,
                                                      inverse=inverse)

    output_channel_size_with_scalar = output_channel_size
    if scalar_term:
        output_channel_size_with_scalar += 1

    if stream:
        signature = empty((output_stream_size, batch_size, output_channel_size_with_scalar))
    else:
        signature = empty((batch_size, output_channel_size_with_scalar))

    if scalar_term:
        # CHANNEL_DIM is last dimension, thus
        signature[:, :, 0] = 1
        signature_with_scalar = signature
        signature = signature.narrow(dim=CHANNEL_DIM, start=1, length=output_channel_size)
    else:
        signature_with_scalar = signature

    signature_by_term: List[Tensor] = []
    signature_by_term_at_stream: List[Tensor] = []

    if stream:
        first_term = signature[0]
        slice_by_term(signature, signature_by_term, input_channel_size, depth)
    else:
        first_term = signature
    slice_by_term(first_term, signature_by_term_at_stream, input_channel_size, depth)

    if initial:
        first_term.copy_(initial_value)
        mult_fused_restricted_exp(path_increments[0],
                                  signature_by_term_at_stream,
                                  inverse=inverse,
                                  reciprocals=reciprocals,
                                  batch_threads=1)
    else:
        restricted_exp(path_increments[0],
                       signature_by_term_at_stream,
                       reciprocals)

    # TODO: parallelism might be done here
    parallelism = False
    batch_threads = 1
    stream_threads = 1

    if parallelism and (not path.is_cuda):
        # implementation should be done here, and modify batch_threads
        pass 

    if stream_threads == 1:
        signature_forward_inner(path_increments,
                                reciprocals,
                                signature,
                                signature_by_term,
                                signature_by_term_at_stream, inverse, stream, start=1,
                                end=output_stream_size, batch_threads=batch_threads)
    else:
        raise NotImplementedError('Multi-threading not possible in python, should implement multiprocessing')

    return signature_with_scalar, path_increments


def signature_backward(grad_signature: Tensor,
                       signature: Tensor,
                       path_increments: Tensor,
                       depth: int,
                       stream: bool,
                       basepoint: bool,
                       inverse: bool,
                       initial: bool,
                       scalar_term: bool):
    if scalar_term:
        grad_signature = grad_signature.narrow(dim=CHANNEL_DIM, start=1,
                                               length=grad_signature.size(CHANNEL_DIM) - 1)
        signature = signature.narrow(dim=CHANNEL_DIM, start=1, length=signature.size(CHANNEL_DIM) - 1)
    grad_signature = grad_signature.detach()
    signature = signature.detach()
    path_increments = path_increments.detach()

    reciprocals = make_reciprocals(depth)
    output_stream_size = path_increments.size(STREAM_DIM)
    input_channel_size = path_increments.size(CHANNEL_DIM)

    signature_by_term: List[Tensor] = []
    slice_by_term(signature, signature_by_term, input_channel_size, depth)

    grad_signature_by_term_at_stream: List[Tensor] = []
    signature_by_term_at_stream: List[Tensor] = []

    if stream:
        grad_signature_at_stream = grad_signature[-1]
    else:
        grad_signature_at_stream = grad_signature

    if scalar_term and initial:
        grad_initial_value = empty((grad_signature_at_stream.size(0),
                                    1 + grad_signature_at_stream.size(1)))
        grad_initial_value.narrow(dim=CHANNEL_DIM, start=0, length=1).zero_()
        grad_initial_value[:, :, 1:(grad_initial_value.size(CHANNEL_DIM))] = grad_signature_at_stream
        grad_signature_at_stream = grad_initial_value.narrow(dim=CHANNEL_DIM, start=1, length=grad_initial_value.size(CHANNEL_DIM) - 1)
    else:
        grad_signature_at_stream = grad_signature_at_stream.clone()
        grad_initial_value = grad_signature_at_stream

    slice_by_term(grad_signature_at_stream, grad_signature_by_term_at_stream, input_channel_size, depth)

    if stream:
        if output_stream_size < 2:
            slice_at_stream(signature_by_term, signature_by_term_at_stream, 0)
    else:
        slice_by_term(signature.clone(), signature_by_term_at_stream, input_channel_size, depth)

    grad_path_increments: Tensor = empty_like(path_increments)

    for stream_index in range(output_stream_size - 1, 0, -1):
        grad_next: Tensor = grad_path_increments[stream_index]
        next_tensor: Tensor = path_increments[stream_index]

        if stream:
            slice_at_stream(signature_by_term, signature_by_term_at_stream, stream_index - 1)
        else:
            mult_fused_restricted_exp(-next_tensor, signature_by_term_at_stream, inverse, reciprocals)

        mult_fused_restricted_exp_backward(next_grad=grad_next,
                                           previous_grad=grad_signature_by_term_at_stream,
                                           next_tensor=next_tensor,
                                           previous=signature_by_term_at_stream,
                                           inverse=inverse,
                                           reciprocals=reciprocals)

        if stream:
            grad_signature_at_stream += grad_signature[stream_index - 1]

    grad_next: Tensor = grad_path_increments[0]
    next_tensor = path_increments[0]

    if initial:
        if stream:
            signature_by_term_at_stream = [el.clone() for el in signature_by_term_at_stream]
        mult_fused_restricted_exp(-next_tensor, signature_by_term_at_stream, inverse, reciprocals)
        mult_fused_restricted_exp_backward(grad_next, grad_signature_by_term_at_stream, next_tensor,
                                           signature_by_term_at_stream, inverse, reciprocals)
    else:
        restricted_exp_backward(grad_next, grad_signature_by_term_at_stream, next_tensor,
                                signature_by_term_at_stream, reciprocals)

    grad_path, grad_basepoint_value = compute_path_increments_backward(grad_path_increments,
                                                                       basepoint,
                                                                       inverse)
    return grad_path, grad_basepoint_value, grad_initial_value
