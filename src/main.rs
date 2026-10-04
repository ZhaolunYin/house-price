use std::error::Error;
use std::io::{self, Write};

mod logic;
use crate::logic::Datum;
use crate::logic::DataNormalized;
use crate::logic::Neuron;

fn test(neuron: &Neuron, data: &DataNormalized) {
    println!("Enter house information:");

    let mut inputs = Vec::new();

    let names = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "floors",
        "condition (1-5)",
        "grade (1-13)",
    ];

    for name in names {
        print!("{name}: ");
        io::stdout().flush().unwrap();

        let mut value = String::new();
        io::stdin().read_line(&mut value).unwrap();

        inputs.push(value.trim().parse::<f64>().unwrap());
    }
    for i in 0..inputs.len() {
        inputs[i] = (inputs[i] - data.means[i]) / data.std_ds[i];
    }

    let data = Datum {
        inputs,
        output: 0.0,
    };

    let prediction = logic::neuron(&neuron, &data);

    println!();
    println!("Predicted price: ${:.2}", prediction);
}

fn normalize(data: &mut DataNormalized) {
    // Normalize inputss:
    let n = data.data.len() as f64;
    let cols = data.data[0].inputs.len();
    data.means = vec![0.0; cols];
    data.std_ds = vec![0.0; cols];

    for d in &data.data {
        for i in 0..cols {
            data.means[i] += d.inputs[i];
        }
    }
    for i in 0..cols {
        data.means[i] /= n;
    }

    for d in &data.data {
        for i in 0..cols {
            let diff = d.inputs[i] - data.means[i];
            data.std_ds[i] += diff * diff;
        }
    }
    for i in 0..cols {
        data.std_ds[i] /= n;
        data.std_ds[i] = data.std_ds[i].sqrt();
    }

    for d in &mut data.data {
        for i in 0..cols {
            d.inputs[i] = (d.inputs[i] - data.means[i]) / data.std_ds[i];
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut data = DataNormalized {
        data: Vec::new(),
        means: Vec::new(),
        std_ds: Vec::new()
    };
    let rdr = csv::Reader::from_path("kc_house_data.csv");
    for result in rdr?.records() {
        let result = result?;
        data.data.push(
            Datum {
                inputs: vec![
                    result[3].parse()?,   // bedrooms
                    result[4].parse()?,   // bathrooms
                    result[5].parse()?,   // sqft_living
                    result[7].parse()?,   // floors
                    result[10].parse()?,  // condition (1-5)
                    result[11].parse()?,  // grade (1-13)
                ],
                output: result[2].parse()?,  // price
            }
        )
    }

    normalize(&mut data);

    let mut neuron = Neuron {
        weights: vec![0.0; data.data[0].inputs.len()],
        bias: 0.0,
    };

    let learning_rate = 0.1;

    let mut prev_loss = logic::loss(&neuron, &data);
    for i in 0..1000 {
        logic::train(&mut neuron, &data, learning_rate);

        if i % 10 == 0 {
            let loss = logic::loss(&neuron, &data);
            println!("iteration {i:6}, loss = {:8}, improvement = {:8}", loss, prev_loss - loss);
            prev_loss = loss;
        }
    }
    print!("weights = ");
    for weight in &neuron.weights {
        print!("{weight} ");
    }
    println!();
    test(&neuron, &data);
    Ok(())
}
