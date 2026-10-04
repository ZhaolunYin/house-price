use rand_distr::{Distribution, Normal};

pub struct Datum {
    pub inputs: Vec<f64>,
    pub output: f64,
}

pub struct DataNormalized {
    pub data: Vec<Datum>,
    pub means: Vec<f64>,
    pub std_ds: Vec<f64>,
}

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub relu: bool,
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(sizes: &[usize]) -> Network {
        assert!(sizes.len() >= 2);
        let layers = sizes.len() - 1;
        let mut network = Network { layers: Vec::new() };

        let mut rng = rand::rng();

        for i in 0..layers {
            let mut layer = Layer {
                neurons: Vec::new(),
                relu: i != layers - 1,
            };

            let std = (2.0 / sizes[i] as f64).sqrt();
            let normal_dist = Normal::new(0.0, std).unwrap();

            for _ in 0..sizes[i + 1] {
                layer.neurons.push(
                    Neuron { 
                        weights: (0..sizes[i]).map(|_| normal_dist.sample(&mut rng)).collect(),
                        bias: 0.0,
                    }
                );
            }
            network.layers.push(layer);
        }
        network
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn neuron(n: &Neuron, d: &Datum) -> f64 {
    let mut sum = n.bias;
    assert_eq!(n.weights.len(), d.inputs.len());
    for i in 0..n.weights.len() {
        sum += n.weights[i] * d.inputs[i];
    }
    sum
}

// Mean squared error
pub fn loss(n: &Neuron, data: &DataNormalized) -> f64 {
    let mut sum = 0.0;
    for d in &data.data {
        // 3x + 1
        let prediction = neuron(n, &d);
        let error = prediction - d.output;
        sum += error * error;
    }
    sum / (data.data.len() as f64)
}

pub fn train(n: &mut Neuron, data: &DataNormalized, learning_rate: f64) {
    let mut gradients = vec![0.0; n.weights.len()];
    let mut bias_grad = 0.0;
    for d in &data.data {
        let prediction = neuron(n, &d);
        let miss = prediction - d.output;
        for i in 0..d.inputs.len() {
            gradients[i] += 2.0 * miss * d.inputs[i];
        }
        bias_grad += 2.0 * miss;
    }
    let l = data.data.len() as f64;
    for i in 0..n.weights.len() {
        gradients[i] /= l;
        n.weights[i] -= learning_rate * gradients[i];
    }
    bias_grad /= l;
    n.bias -= learning_rate * bias_grad;
}
