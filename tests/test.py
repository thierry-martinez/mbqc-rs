import functools
import hypothesis as hyp
import numpy as np
import mbqc_rs

atol = 1e-8


def get_nqubits(array: np.ndarray) -> int:
    return len(array).bit_length() - 1


def reshape_tensor(array: np.ndarray) -> np.ndarray:
    return array.reshape((2,) * get_nqubits(array))


def get_norm(array: np.ndarray) -> float:
    return np.sqrt(np.sum(array.flatten().conj() * array.flatten()))


def is_power_of_two(n: int) -> bool:
    return n & (n - 1) == 0


complex_st = hyp.strategies.complex_numbers(min_magnitude=1e-5, max_magnitude=1e5)


def array_st(min_length=0, max_length=8):
    return (
        hyp.strategies.integers(min_value=min_length, max_value=max_length)
        .flatmap(lambda nqubits: hyp.strategies.lists(complex_st, min_size=1 << nqubits, max_size=1 << nqubits))
        .map(lambda l: np.array(l, dtype=np.complex128))
    )


def non_null_array_st(min_length=0, max_length=8):
    return array_st(min_length, max_length).filter(lambda array: get_norm(array) != 0)


@hyp.given(
    hyp.strategies.integers(min_value=0, max_value=16),
    hyp.strategies.sampled_from([mbqc_rs.Zero, mbqc_rs.Plus]),
)
def test_new_vec(nqubits, state):
    vec = mbqc_rs.new_vec(nqubits, state)
    assert mbqc_rs.get_nqubits(vec) == nqubits
    array = mbqc_rs.get_vec(vec)
    assert len(array) == 1 << nqubits
    if state == mbqc_rs.Zero:
        state_mat = np.array([1, 0])
    elif state == mbqc_rs.Plus:
        state_mat = np.array([1, 1]) / np.sqrt(2)
    else:
        assert False
    ref = functools.reduce(np.kron, (state_mat for _ in range(nqubits)), np.array(1, dtype=np.complex128))
    np.testing.assert_allclose(array, ref.flatten())


@hyp.given(array_st())
def test_from_vec(array):
    nqubits = get_nqubits(array)
    norm = get_norm(array)
    try:
        vec = mbqc_rs.from_vec(array)
        assert norm != 0
    except ValueError:
        assert norm == 0
        return
    assert mbqc_rs.get_nqubits(vec) == nqubits
    array2 = mbqc_rs.get_vec(vec)
    assert len(array2) == 1 << nqubits
    array /= norm
    np.testing.assert_allclose(array, array2)


@hyp.given(hyp.strategies.lists(complex_st).map(np.array))
def test_from_vec_invalid_size(array):
    try:
        vec = mbqc_rs.from_vec(array)
        valid = True
    except TypeError:
        assert len(array) == 0
        return
    except ValueError:
        valid = False
    assert valid == (is_power_of_two(len(array)) and get_norm(array) != 0)


@hyp.given(non_null_array_st(), non_null_array_st())
def test_tensor_array(a, b):
    result = mbqc_rs.tensor_array(a, b)
    ref = np.kron(reshape_tensor(a), reshape_tensor(b)).flatten()
    np.testing.assert_allclose(result, ref)


@hyp.given(non_null_array_st(), non_null_array_st())
def test_tensor(a, b):
    vec = mbqc_rs.from_vec(a)
    mbqc_rs.tensor(vec, b)
    result = mbqc_rs.get_vec(vec)
    a /= get_norm(a)
    ref = np.kron(reshape_tensor(a), reshape_tensor(b)).flatten()
    np.testing.assert_allclose(result, ref)


# @hyp.given(non_null_array_st().flatmap(lambda array: hyp.strategies.integers(min_value=0, max_value=min(get_nqubits(array), 10)).flatmap(lambda nqubits: hyp.strategies.lists(complex_st, min_size=1 << (2 * nqubits), max_size=1 << (2 * nqubits)).flatmap(lambda tensor:  hyp.strategies.permutations(list(range(get_nqubits(array)))).map(lambda qubits: (array, tensor, qubits[0:nqubits]))))))
# def test_apply(params):
#    (array, tensor, qubits) = params
#    tensor = np.array(tensor)
#    vec_array = mbqc_rs.from_vec(array)
#    mbqc_rs.apply(vec_array, tensor, qubits)
#    result = mbqc_rs.get_vec(vec_array)
#    tensor_qubits = tuple(range(len(qubits), 2 * len(qubits)))
#    base_qubits = tuple(range(len(qubits)))
#    array /= get_norm(array)
#    array = np.tensordot(reshape_tensor(tensor), reshape_tensor(array), (tensor_qubits, qubits))
#    array = np.moveaxis(array, base_qubits, qubits)
#    np.testing.assert_allclose(result, array.flatten())
#


def array_and_one_qubit_st():
    return (
        non_null_array_st()
        .filter(lambda array: len(array) >= 2)
        .flatmap(
            lambda array: hyp.strategies.integers(min_value=0, max_value=get_nqubits(array) - 1).map(
                lambda qubit: (array, qubit)
            )
        )
    )


@hyp.given(array_and_one_qubit_st(), non_null_array_st(min_length=2, max_length=2))
def test_evolve(pair, op):
    (array, qubit) = pair
    vec_array = mbqc_rs.from_vec(array)
    mbqc_rs.evolve(vec_array, op, qubit)
    result = mbqc_rs.get_vec(vec_array)
    array /= get_norm(array)
    array = np.tensordot(reshape_tensor(op), reshape_tensor(array), (1, qubit))
    array = np.moveaxis(array, 0, qubit)
    np.testing.assert_allclose(result, array.flatten(), atol=atol)


@hyp.given(array_and_one_qubit_st(), non_null_array_st(min_length=2, max_length=2))
def test_expectation_value(pair, op):
    (array, qubit) = pair
    vec_array = mbqc_rs.from_vec(array)
    result = mbqc_rs.expectation_value(vec_array, op, qubit)
    array /= get_norm(array)
    array = reshape_tensor(array)
    evolved = np.tensordot(reshape_tensor(op), array, (1, qubit))
    evolved = np.moveaxis(evolved, 0, qubit)
    expected = np.dot(array.flatten().conj(), evolved.flatten())
    assert np.abs(result - expected) < atol


CZ_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]],
    dtype=np.complex128,
)
CNOT_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]],
    dtype=np.complex128,
)
SWAP_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]],
    dtype=np.complex128,
)


def array_and_two_qubits_st():
    return (
        non_null_array_st()
        .filter(lambda array: len(array) >= 4)
        .flatmap(
            lambda array: hyp.strategies.permutations(list(range(get_nqubits(array)))).map(
                lambda qubits: (array, tuple(qubits[0:2]))
            )
        )
    )


@hyp.given(
    array_and_two_qubits_st(),
    hyp.strategies.sampled_from(
        [
            (CZ_TENSOR, mbqc_rs.entangle),
            (SWAP_TENSOR, mbqc_rs.swap),
            (CNOT_TENSOR, lambda vec, pair: mbqc_rs.cnot(vec, pair[0], pair[1])),
        ]
    ),
)
def test_operator(array_pair, op_pair):
    (array, qubits) = array_pair
    (tensor, method) = op_pair
    vec_array = mbqc_rs.from_vec(array)
    method(vec_array, qubits)
    result = mbqc_rs.get_vec(vec_array)
    array /= get_norm(array)
    array = reshape_tensor(array)
    expected = np.tensordot(tensor, array, ((2, 3), qubits))
    expected = np.moveaxis(expected, (0, 1), qubits)
    np.testing.assert_allclose(result, expected.flatten(), atol=atol)
