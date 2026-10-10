use rand_distr::{Distribution, Normal};

pub struct Datum {
    pub inputs: Vec<f64>,
    pub output: f64,
}

pub struct DataNormalized {
    pub data: Vec<Datum>,

    pub means: Vec<f64>,
    pub stds: Vec<f64>,

    pub out_mean: f64,
    pub out_std: f64,
}

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,

    pub weight_grad: Vec<f64>,
    pub bias_grad: f64,
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub relu: bool,

    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
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
                inputs: Vec::new(),
                outputs: Vec::new(),
            };

            let std = (2.0 / sizes[i] as f64).sqrt();
            let normal_dist = Normal::new(0.0, std).unwrap();

            for _ in 0..sizes[i + 1] {
                layer.neurons.push(
                    Neuron { 
                        weights: (0..sizes[i]).map(|_| normal_dist.sample(&mut rng)).collect(),
                        bias: 0.0,

                        weight_grad: vec![0.0; sizes[i]],
                        bias_grad: 0.0,
                    }
                );
            }
            network.layers.push(layer);
        }
        network
    }
    pub fn predict(&mut self, inputs: &[f64], save: bool) -> Vec<f64> {
        let mut output = Vec::from(inputs);
        for i in 0..self.layers.len() {
            if save {
                self.layers[i].inputs = output.clone();
            }
            output = layer_forward(&self.layers[i], &output);
            if save {
                self.layers[i].outputs = output.clone();
            }
        }
        output
    }
    pub fn train(&mut self, data: &DataNormalized, learning_rate: f64) -> f64 {
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                neuron.weight_grad.fill(0.0);
                neuron.bias_grad = 0.0;
            }
        }
        let mut total_loss = 0.0;
        for d in &data.data {
            let prediction = self.predict(&d.inputs, true);
            let miss = prediction[0] - d.output;
            total_loss += miss * miss;
            let mut blame = vec![2.0 * miss];
            for layer in self.layers.iter_mut().rev() {
                let mut prev_blame = vec![0.0; layer.inputs.len()];
                for (j, n) in layer.neurons.iter_mut().enumerate() {
                    let delta = if layer.relu && layer.outputs[j] == 0.0 {
                        0.0
                    }
                    else {
                        blame[j]
                    };
                    n.bias_grad += delta;
                    for k in 0..n.weight_grad.len() {
                        n.weight_grad[k] += delta * layer.inputs[k];
                    }
                    for k in 0..n.weights.len() {
                        prev_blame[k] += delta * n.weights[k]
                    }
                }
                blame = prev_blame;
            }
        }
        let l = data.data.len() as f64;
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                assert_eq!(neuron.weights.len(), neuron.weight_grad.len());
                for i in 0..neuron.weights.len() {
                    neuron.weight_grad[i] /= l;
                    neuron.weights[i] -= learning_rate * neuron.weight_grad[i]
                }
                neuron.bias_grad /= l;
                neuron.bias -= learning_rate * neuron.bias_grad;
            }
        }
        total_loss / l
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn neuron_forward(n: &Neuron, inputs: &[f64]) -> f64 {
    let mut sum = n.bias;
    assert_eq!(n.weights.len(), inputs.len());
    for i in 0..n.weights.len() {
        sum += n.weights[i] * inputs[i];
    }
    sum
}

fn layer_forward(l: &Layer, inputs: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    for n in &l.neurons {
        let output = neuron_forward(&n, inputs);
        result.push(if l.relu {
            relu(output)
        }
        else {
            output
        });
    }
    result
}

// Mean squared error
pub fn loss(n: &Neuron, data: &DataNormalized) -> f64 {
    let mut sum = 0.0;
    for d in &data.data {
        // 3x + 1
        let prediction = neuron_forward(n, &d.inputs);
        let error = prediction - d.output;
        sum += error * error;
    }
    sum / (data.data.len() as f64)
}

pub fn train(n: &mut Neuron, data: &DataNormalized, learning_rate: f64) {
    let mut gradients = vec![0.0; n.weights.len()];
    let mut bias_grad = 0.0;
    for d in &data.data {
        let prediction = neuron_forward(n, &d.inputs);
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
