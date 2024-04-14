use num_complex;
use pyo3;
use pyo3::prelude::PyCapsuleMethods;
use rayon::prelude::*;

#[pyo3::pyclass]
#[derive(Copy, Clone)]
pub enum InitialState {
    Zero,
    Plus,
}

type Scalar = num_complex::Complex<f64>;

struct Statevec {
    nqubits: u8,
    psi: Vec<Scalar>,
    cache: Vec<Scalar>,
}

type Op = [Scalar; 4];

const EPSILON: f64 = 1e-300;

pub fn is_close(a: f64, b: f64) -> bool {
    return (a - b).abs() < EPSILON;
}

const C0: Scalar = Scalar::new(0., 0.);

const C1: Scalar = Scalar::new(1., 0.);

pub fn get_norm<Item: std::borrow::Borrow<Scalar>>(
    it: impl std::iter::Iterator<Item = Item>,
) -> f64 {
    it.map(|v| v.borrow().norm_sqr()).sum::<f64>().sqrt()
}

pub fn is_power_of_two(n: usize) -> bool {
    n & (n - 1) == 0
}

pub fn tensor<'a>(
    a: &'a [Scalar],
    b: &'a [Scalar],
) -> impl std::iter::ExactSizeIterator<Item = Scalar> + 'a {
    debug_assert!(is_power_of_two(a.len()));
    debug_assert!(is_power_of_two(b.len()));
    let a_nqubits = a.len().ilog2() as u8;
    let b_nqubits = b.len().ilog2() as u8;
    let nqubits = a_nqubits.checked_add(b_nqubits).unwrap();
    let size: usize = 1 << nqubits;
    (0..size).map(move |index| {
        let mut a_index = 0;
        let mut b_index = 0;
        let mut index_left = index;
        for i in 0..std::cmp::min(a_nqubits, b_nqubits) {
            if index_left & 1 != 0 {
                b_index |= 1 << i;
            }
            if index_left & 0b10 != 0 {
                a_index |= 1 << i;
            }
            index_left >>= 2;
        }
        if a_nqubits < b_nqubits {
            b_index |= index_left << a_nqubits;
        } else if a_nqubits > b_nqubits {
            a_index |= index_left << b_nqubits;
        }
        a[a_index] * b[b_index]
    })
}

pub fn take<'a>(
    psi: &'a [Scalar],
    qubit: u8,
    index: bool,
) -> impl std::iter::ExactSizeIterator<Item = Scalar> + 'a {
    debug_assert!(is_power_of_two(psi.len()));
    let nqubits = psi.len().ilog2() as u8;
    let size: usize = 1 << (nqubits - 1);
    let bit = 1 << (nqubits - qubit - 1);
    let value = if index { bit } else { 0 };
    let mask = bit - 1;
    (0..size).map(move |index| {
        let psi_index = ((index & !mask) << 1) | value | index & mask;
        psi[psi_index]
    })
}

pub fn initial_vector(nqubits: u8, initial_state: InitialState) -> Vec<Scalar> {
    let size: usize = 1 << nqubits;
    match initial_state {
        InitialState::Zero => {
            let mut psi: Vec<Scalar> = vec![C0; size];
            psi[0] = C1;
            psi
        }
        InitialState::Plus => {
            let initial_value: Scalar = (1. / f64::powf(2., nqubits as f64 / 2.)).into();
            vec![initial_value; size]
        }
    }
}

struct TwoBits {
    low: u8,
    high: u8,
    mask_low: usize,
    mask_high: usize,
}

impl TwoBits {
    pub fn new(nqubits: u8, qubits: (u8, u8)) -> Self {
        let (low, high) = if qubits.0 < qubits.1 {
            (nqubits - qubits.1 - 1, nqubits - qubits.0 - 1)
        } else {
            (nqubits - qubits.0 - 1, nqubits - qubits.1 - 1)
        };
        let mask_low = (1 << low) - 1;
        let mask_high = (1 << (high - 1)) - 1;
        Self {
            low,
            high,
            mask_low,
            mask_high,
        }
    }

    pub fn value(self: &Self, base: usize, values: (bool, bool)) -> usize {
        (base & !self.mask_high) << 2
            | ((values.0 as usize) << self.high)
            | (base & self.mask_high & !self.mask_low) << 1
            | ((values.1 as usize) << self.low)
            | base & self.mask_low
    }
}

struct OrderedTwoBits {
    two_bits: TwoBits,
    swapped: bool,
}

impl OrderedTwoBits {
    pub fn new(nqubits: u8, qubits: (u8, u8)) -> Self {
        Self {
            two_bits: TwoBits::new(nqubits, qubits),
            swapped: qubits.0 > qubits.1,
        }
    }

    pub fn value(self: &Self, base: usize, values: (bool, bool)) -> usize {
        let values = if self.swapped {
            (values.1, values.0)
        } else {
            values
        };
        self.two_bits.value(base, values)
    }
}

/*
enum Side {
    Zero,
    Left(Scalar),
    Right(Scalar),
    Both { left: Scalar, right: Scalar }
}

impl Side {
    pub fn new(left: Scalar, right: Scalar) -> Self {
        if is_close(left.norm(), 0.) {
            if is_close(right.norm(), 0.) {
                Side::Zero
            }
            else {
                Side::Right(right)
            }
        }
        else if is_close(right.norm(), 0.) {
            Side::Left(left)
        }
        else {
            Side::Both { left, right }
        }
    }
}
*/

impl Statevec {
    pub fn new(nqubits: u8, initial_state: InitialState) -> Self {
        let psi = initial_vector(nqubits, initial_state);
        Self {
            nqubits,
            psi,
            cache: vec![],
        }
    }

    pub fn from_vec(mut psi: Vec<Scalar>) -> Result<Self, String> {
        let size = psi.len();
        if !is_power_of_two(size) {
            return Err(format!("from_vec() applied on a vector of length {size}, whereas a power of two is required"));
        }
        let nqubits = size.ilog2() as u8;
        let norm = get_norm(psi.iter());
        if is_close(norm, 0.) {
            return Err("from_vec() applied on a null vector".into());
        }
        for cell in psi.iter_mut() {
            *cell /= norm;
        }
        Ok(Self {
            nqubits,
            psi,
            cache: vec![],
        })
    }

    pub fn apply(&mut self, tensor: &[Scalar], qubits: &[u8]) {
        debug_assert!(tensor.len() == 1 << (2 * qubits.len()));
        let nqubits = self.nqubits;
        debug_assert!(self.psi.len() == 1 << nqubits);
        let sum_len = 1 << qubits.len();
        self.cache.clear();
        self.cache
            .par_extend((0..1 << nqubits).into_par_iter().map(|index| {
                let mut sum = C0;
                let mut index_tensor_high = 0;
                for (qubit_index, qubit) in qubits.iter().enumerate() {
                    if index & (1 << (nqubits - qubit - 1)) != 0 {
                        index_tensor_high |= 1 << (2 * qubits.len() - qubit_index - 1)
                    }
                }
                let index_tensor_high = index_tensor_high;
                for k in 0..sum_len {
                    let index_tensor = index_tensor_high | k;
                    let mut index_source: usize = index as usize;
                    let mut left = k;
                    for qubit in qubits.iter().rev() {
                        let mask = 1 << (nqubits - qubit - 1);
                        if left & 1 == 0 {
                            index_source &= !mask
                        } else {
                            index_source |= mask
                        }
                        left >>= 1;
                    }
                    sum += tensor[index_tensor] * self.psi[index_source];
                }
                sum
            }));
        std::mem::swap(&mut self.psi, &mut self.cache);
    }

    pub fn entangle(&mut self, qubits: (u8, u8)) {
        let two_bits = TwoBits::new(self.nqubits, qubits);
        for index in 0..1 << (self.nqubits - 2) {
            let full_index = two_bits.value(index, (true, true));
            self.psi[full_index] = -self.psi[full_index];
        }
    }

    pub fn swap(&mut self, qubits: (u8, u8)) {
        let two_bits = TwoBits::new(self.nqubits, qubits);
        for index in 0..1 << (self.nqubits - 2) {
            let index0 = two_bits.value(index, (true, false));
            let index1 = two_bits.value(index, (false, true));
            let tmp = self.psi[index0];
            self.psi[index0] = self.psi[index1];
            self.psi[index1] = tmp;
        }
    }

    pub fn cnot(&mut self, control: u8, target: u8) {
        let two_bits = OrderedTwoBits::new(self.nqubits, (control, target));
        for index in 0..1 << (self.nqubits - 2) {
            let index0 = two_bits.value(index, (true, false));
            let index1 = two_bits.value(index, (true, true));
            let tmp = self.psi[index0];
            self.psi[index0] = self.psi[index1];
            self.psi[index1] = tmp;
        }
    }

    pub fn pre_evolve(&mut self, op: &Op, qubit: u8) {
        let mask: usize = 1 << (self.nqubits - qubit - 1);
        self.cache.clear();
        /*
                let lines: [Side; 2] = [Side::new(op[0], op[1]), Side::new(op[2], op[3])];
                self.cache.par_extend((0 .. 1 << self.nqubits).into_par_iter().map(|index| {
                    match lines[(index & mask != 0) as usize] {
                        Side::Zero => C0,
                        Side::Left(left) => {
                            let index_cell0: usize = index & !mask;
                            left * self.psi[index_cell0]
                        },
                        Side::Right(right) => {
                            let index_cell1: usize = index | mask;
                            right * self.psi[index_cell1]
                        },
                        Side::Both { left, right } => {
                            let index_cell0: usize = index & !mask;
                            let index_cell1: usize = index | mask;
                            left * self.psi[index_cell0] + right * self.psi[index_cell1]
                        },
                    }
                }));
        */
        self.cache
            .par_extend((0..1 << self.nqubits).into_par_iter().map(|index| {
                let index_op_high = if index & mask != 0 { 2 } else { 0 };
                let index_cell0: usize = index & !mask;
                let index_cell1: usize = index | mask;
                op[index_op_high] * self.psi[index_cell0]
                    + op[index_op_high | 1] * self.psi[index_cell1]
            }));
    }

    pub fn evolve(&mut self, op: &Op, qubit: u8) {
        self.pre_evolve(op, qubit);
        std::mem::swap(&mut self.psi, &mut self.cache);
    }

    pub fn expectation_value(&mut self, op: &Op, qubit: u8) -> Scalar {
        self.pre_evolve(op, qubit);
        // self.psi.iter().zip(self.cache.iter()).map(|(s, t)| s.conj() * t).sum()
        (0..1 << self.nqubits)
            .into_par_iter()
            .map(|index| self.psi[index].conj() * self.cache[index])
            .sum()
    }

    pub fn tensor(&mut self, other: &[Scalar]) {
        debug_assert!(is_power_of_two(other.len()));
        let other_nqubits = other.len().ilog2() as u8;
        self.nqubits = self.nqubits + other_nqubits;
        self.cache.clear();
        self.cache.extend(tensor(self.psi.as_slice(), other));
        std::mem::swap(&mut self.psi, &mut self.cache);
    }

    pub fn remove_qubit(&mut self, qubit: u8) {
        self.nqubits = self.nqubits - 1;
        let norm = get_norm(take(&self.psi, qubit, false));
        self.cache.clear();
        if is_close(norm, 0.) {
            self.cache.extend(take(&self.psi, qubit, true));
            let norm = get_norm(self.cache.iter());
            for cell in self.cache.iter_mut() {
                *cell /= norm;
            }
        } else {
            self.cache
                .extend(take(&self.psi, qubit, false).map(|v| v / norm))
        };
        std::mem::swap(&mut self.psi, &mut self.cache);
    }
}

#[pyo3::pymodule]
fn mbqc_rs<'py>(
    _py: pyo3::prelude::Python<'py>,
    m: &pyo3::prelude::Bound<'py, pyo3::types::PyModule>,
) -> pyo3::prelude::PyResult<()> {
    m.add("Zero", InitialState::Zero)?;
    m.add("Plus", InitialState::Plus)?;

    type PyVec<'py> = pyo3::prelude::Bound<'py, pyo3::types::PyCapsule>;

    fn make_pyvec<'py>(
        py: pyo3::prelude::Python<'py>,
        vec: Statevec,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        let capsule_name = std::ffi::CString::new("statevec").unwrap();
        pyo3::types::PyCapsule::new_bound(py, vec, Some(capsule_name))
    }

    fn get_pyvec<'py>(vec: PyVec<'py>) -> &Statevec {
        unsafe { vec.reference::<Statevec>() }
    }

    fn get_pyvec_mut<'py>(vec: PyVec<'py>) -> &mut Statevec {
        unsafe { &mut *vec.pointer().cast() }
    }

    #[pyo3::pyfunction]
    fn new_vec<'py>(
        py: pyo3::prelude::Python<'py>,
        nqubits: u8,
        initial_state: InitialState,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        make_pyvec(py, Statevec::new(nqubits, initial_state))
    }
    m.add_function(pyo3::wrap_pyfunction!(new_vec, m)?)?;

    #[pyo3::pyfunction]
    fn from_vec<'py>(
        py: pyo3::prelude::Python<'py>,
        vec: numpy::borrow::PyReadonlyArrayDyn<Scalar>,
    ) -> pyo3::prelude::PyResult<PyVec<'py>> {
        make_pyvec(
            py,
            Statevec::from_vec(vec.as_slice()?.to_vec())
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        )
    }
    m.add_function(pyo3::wrap_pyfunction!(from_vec, m)?)?;

    #[pyo3::pyfunction]
    fn get_nqubits<'py>(vec: PyVec<'py>) -> pyo3::prelude::PyResult<u8> {
        Ok(get_pyvec(vec).nqubits)
    }
    m.add_function(pyo3::wrap_pyfunction!(get_nqubits, m)?)?;

    #[pyo3::pyfunction]
    fn get_vec<'py>(
        py: pyo3::prelude::Python<'py>,
        py_vec: PyVec<'py>,
    ) -> pyo3::prelude::Bound<'py, numpy::array::PyArray1<Scalar>> {
        let vec = get_pyvec(py_vec);
        numpy::IntoPyArray::into_pyarray_bound(vec.psi.to_vec(), py)
    }
    m.add_function(pyo3::wrap_pyfunction!(get_vec, m)?)?;

    #[pyo3::pyfunction]
    fn tensor_array<'py>(
        py: pyo3::prelude::Python<'py>,
        py_a: numpy::borrow::PyReadonlyArrayDyn<Scalar>,
        py_b: numpy::borrow::PyReadonlyArrayDyn<Scalar>,
    ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::array::PyArray1<Scalar>>> {
        Ok(numpy::array::PyArray1::from_iter_bound(
            py,
            tensor(py_a.as_slice()?, py_b.as_slice()?),
        ))
    }
    m.add_function(pyo3::wrap_pyfunction!(tensor_array, m)?)?;

    #[pyo3::pyfunction]
    #[pyo3(name = "tensor")]
    fn tensor_py<'py>(
        py_vec: PyVec<'py>,
        array: numpy::borrow::PyReadonlyArrayDyn<Scalar>,
    ) -> pyo3::prelude::PyResult<()> {
        let vec = get_pyvec_mut(py_vec);
        vec.tensor(array.as_slice()?);
        Ok(())
    }
    m.add_function(pyo3::wrap_pyfunction!(tensor_py, m)?)?;

    #[pyo3::pyfunction]
    fn add_nodes<'py>(
        py_vec: PyVec<'py>,
        nqubits: u8,
        initial_state: InitialState,
    ) -> pyo3::prelude::PyResult<()> {
        let vec = get_pyvec_mut(py_vec);
        vec.tensor(initial_vector(nqubits, initial_state).as_slice());
        Ok(())
    }
    m.add_function(pyo3::wrap_pyfunction!(add_nodes, m)?)?;

    #[pyo3::pyfunction]
    fn apply<'py>(
        py_vec: PyVec<'py>,
        py_tensor: numpy::borrow::PyReadonlyArrayDyn<Scalar>,
        qubits: Vec<u8>,
    ) -> pyo3::prelude::PyResult<()> {
        let vec = get_pyvec_mut(py_vec);
        Ok(vec.apply(py_tensor.as_slice()?, &qubits))
    }
    m.add_function(pyo3::wrap_pyfunction!(apply, m)?)?;

    #[pyo3::pyfunction]
    fn entangle<'py>(py_vec: PyVec<'py>, qubits: (u8, u8)) -> pyo3::prelude::PyResult<()> {
        let vec = get_pyvec_mut(py_vec);
        Ok(vec.entangle(qubits))
    }
    m.add_function(pyo3::wrap_pyfunction!(entangle, m)?)?;

    #[pyo3::pyfunction]
    fn cnot<'py>(py_vec: PyVec<'py>, control: u8, target: u8) -> pyo3::prelude::PyResult<()> {
        let vec = get_pyvec_mut(py_vec);
        Ok(vec.cnot(control, target))
    }
    m.add_function(pyo3::wrap_pyfunction!(cnot, m)?)?;

    #[pyo3::pyfunction]
    fn swap<'py>(py_vec: PyVec<'py>, qubits: (u8, u8)) -> pyo3::prelude::PyResult<()> {
        let vec = get_pyvec_mut(py_vec);
        Ok(vec.swap(qubits))
    }
    m.add_function(pyo3::wrap_pyfunction!(swap, m)?)?;

    #[pyo3::pyfunction]
    fn evolve<'py>(
        py_vec: PyVec<'py>,
        py_op: numpy::borrow::PyReadonlyArrayDyn<Scalar>,
        qubit: u8,
    ) -> pyo3::prelude::PyResult<()> {
        let vec = get_pyvec_mut(py_vec);
        Ok(vec.evolve(py_op.as_slice()?.try_into()?, qubit))
    }
    m.add_function(pyo3::wrap_pyfunction!(evolve, m)?)?;

    #[pyo3::pyfunction]
    fn expectation_value<'py>(
        py_vec: PyVec<'py>,
        py_op: numpy::borrow::PyReadonlyArrayDyn<Scalar>,
        qubit: u8,
    ) -> pyo3::prelude::PyResult<Scalar> {
        let vec = get_pyvec_mut(py_vec);
        Ok(vec.expectation_value(py_op.as_slice()?.try_into()?, qubit))
    }
    m.add_function(pyo3::wrap_pyfunction!(expectation_value, m)?)?;

    #[pyo3::pyfunction]
    fn remove_qubit<'py>(py_vec: PyVec<'py>, qubit: u8) -> pyo3::prelude::PyResult<()> {
        let vec = get_pyvec_mut(py_vec);
        vec.remove_qubit(qubit);
        Ok(())
    }
    m.add_function(pyo3::wrap_pyfunction!(remove_qubit, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_bits() {
        assert_eq!(TwoBits::new(4, (0, 1)).value(0b0000, (true, true)), 0b1100);
        assert_eq!(TwoBits::new(4, (0, 1)).value(0b0000, (true, false)), 0b1000);
        assert_eq!(TwoBits::new(4, (0, 1)).value(0b0000, (false, true)), 0b0100);
        assert_eq!(
            TwoBits::new(4, (0, 1)).value(0b0000, (false, false)),
            0b0000
        );
        assert_eq!(TwoBits::new(4, (0, 1)).value(0b0010, (true, true)), 0b1110);
        assert_eq!(TwoBits::new(4, (0, 1)).value(0b0001, (true, false)), 0b1001);
        assert_eq!(TwoBits::new(4, (0, 1)).value(0b0010, (false, true)), 0b0110);
        assert_eq!(
            TwoBits::new(4, (0, 1)).value(0b0001, (false, false)),
            0b0001
        );
    }

    #[test]
    fn test_ordered_two_bits() {
        assert_eq!(
            OrderedTwoBits::new(4, (0, 1)).value(0b0000, (true, true)),
            0b1100
        );
        assert_eq!(
            OrderedTwoBits::new(4, (1, 0)).value(0b0000, (false, true)),
            0b1000
        );
        assert_eq!(
            OrderedTwoBits::new(4, (0, 1)).value(0b0000, (false, true)),
            0b0100
        );
        assert_eq!(
            OrderedTwoBits::new(4, (1, 0)).value(0b0000, (false, false)),
            0b0000
        );
        assert_eq!(
            OrderedTwoBits::new(4, (0, 1)).value(0b0010, (true, true)),
            0b1110
        );
        assert_eq!(
            OrderedTwoBits::new(4, (1, 0)).value(0b0001, (false, true)),
            0b1001
        );
        assert_eq!(
            OrderedTwoBits::new(4, (0, 1)).value(0b0010, (false, true)),
            0b0110
        );
        assert_eq!(
            OrderedTwoBits::new(4, (1, 0)).value(0b0001, (false, false)),
            0b0001
        );
    }

    #[test]
    fn test_entangle() {
        let mut vec = Statevec::new(2, InitialState::Plus);
        vec.entangle((0, 1));
        assert_eq!(
            vec.psi,
            vec![
                Scalar::new(0.5, 0.),
                Scalar::new(0.5, 0.),
                Scalar::new(0.5, 0.),
                Scalar::new(-0.5, 0.)
            ]
        );
    }

    #[test]
    fn test_cnot() {
        let mut vec = Statevec::new(2, InitialState::Plus);
        vec.entangle((0, 1));
        vec.cnot(0, 1);
        assert_eq!(
            vec.psi,
            vec![
                Scalar::new(0.5, 0.),
                Scalar::new(0.5, 0.),
                Scalar::new(-0.5, 0.),
                Scalar::new(0.5, 0.)
            ]
        );
    }
}
