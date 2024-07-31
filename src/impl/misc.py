from typing import List

from torch import Tensor, linspace, ones

MAXIMAL_INTEGER_VALUE: int = 2 ** 31 - 1
STREAM_DIM: int = -3
BATCH_DIM: int = -2
CHANNEL_DIM: int = -1


def checkargs_channels_depth(channels: int, depth: int) -> None:
    if channels < 1:
        raise ValueError("Argument 'channels' must be at least one.")
    if depth < 1:
        raise ValueError("Argument 'depth' must be an integer greater than or equal to one.")
    return 


def signature_channels(input_channel_size: int, depth: int, scalar_term: bool) -> int:
    """_summary_

    Parameters
    ----------
    input_channel_size : int
        _description_
    depth : int
        _description_
    scalar_term : bool
        _description_

    Returns
    -------
    int
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    OverflowError
        _description_
    OverflowError
        _description_
    """
    if input_channel_size < 1:
        raise ValueError("input_channels must be at least 1")
    if depth < 1:
        raise ValueError("depth must be at least 1")

    if input_channel_size == 1:
        output_channels: int = depth + 1 if scalar_term else depth
    else:
        output_channels: int = input_channel_size
        mul_limit = MAXIMAL_INTEGER_VALUE / input_channel_size
        add_limit = MAXIMAL_INTEGER_VALUE - input_channel_size

        for _ in range(1, depth):
            if output_channels > mul_limit:
                raise OverflowError("Integer overflow detected.")
            output_channels *= input_channel_size
            if output_channels > add_limit:
                raise OverflowError("Integer overflow detected.")
            output_channels += input_channel_size

        if scalar_term:
            output_channels += 1
    return output_channels


def make_reciprocals(depth: int):
    if depth > 1:
         return linspace(2, depth, depth - 1).reciprocal()
    return ones(0)


# TODO: might be interesting to reserve memory for lists through appending None
# equivalent of .reserve method of std::vector in c++

def slice_by_term(inputs: Tensor, outputs: List[Tensor], input_channel_size: int, depth: int):
    current_memory_pos = 0
    current_memory_length = input_channel_size

    outputs.clear()

    for _ in range(depth):
        outputs.append(inputs.narrow(dim=CHANNEL_DIM, start=current_memory_pos, len=current_memory_length))
        current_memory_pos += current_memory_length
        current_memory_length *= input_channel_size
    return 


def slice_at_stream(inputs: Tensor, outputs: List[Tensor], stream_index: int):
    outputs.clear()
    for element in inputs:
        outputs.append(element[stream_index])
    return 
