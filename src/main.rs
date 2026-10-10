use std::error::Error;
use std::io::{self, Write};

mod logic;
use crate::logic::{Datum, DataNormalized, Network};

fn test(network: &mut Network, data: &DataNormalized) {
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
        inputs[i] = normalize_value(inputs[i], data.means[i], data.stds[i]);
    }
    let prediction = network.predict(&inputs, false);
    let prediction = normalize_value_reverse(prediction[0], data.out_mean, data.out_std);

    println!();
    println!("Predicted price: ${:.2}", prediction);
}

fn normalize_value(v: f64, mean: f64, std: f64) -> f64 {
    (v - mean) / std
}

fn normalize_value_reverse(v: f64, mean: f64, std: f64) -> f64 {
    v * std + mean
}

fn normalize(data: &mut DataNormalized) {
    // Normalize inputss:
    let n = data.data.len() as f64;
    let cols = data.data[0].inputs.len();
    data.means = vec![0.0; cols];
    data.stds = vec![0.0; cols];
    data.out_mean = 0.0;
    data.out_std = 0.0;

    for d in &data.data {
        for i in 0..cols {
            data.means[i] += d.inputs[i];
        }
        data.out_mean += d.output;
    }

    for i in 0..cols {
        data.means[i] /= n;
    }
    data.out_mean /= n;

    for d in &data.data {
        for i in 0..cols {
            let diff = d.inputs[i] - data.means[i];
            data.stds[i] += diff * diff;
        }
        let diff = d.output - data.out_mean;
        data.out_std += diff * diff;
    }

    for i in 0..cols {
        data.stds[i] /= n;
        data.stds[i] = data.stds[i].sqrt();
    }
    data.out_std /= n;
    data.out_std = data.out_std.sqrt();

    for d in &mut data.data {
        for i in 0..cols {
            d.inputs[i] = normalize_value(d.inputs[i], data.means[i], data.stds[i]);
        }
        d.output = normalize_value(d.output, data.out_mean, data.out_std);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut data = DataNormalized {
        data: Vec::new(),
        means: Vec::new(),
        stds: Vec::new(),

        out_mean: 0.0,
        out_std: 0.0,
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

    let mut network = Network::new(&[data.data[0].inputs.len(), 3, 3, 1]);

    let learning_rate = 0.1;

    let mut prev_loss: f64 = f64::INFINITY;
    for i in 0..1000 {
        let loss = network.train(&data, learning_rate);

        if i % 10 == 0 {
            println!("iteration {i:6}, loss = {:8}, improvement = {:8}", loss, prev_loss - loss);
            prev_loss = loss;
        }
    }
    println!();
    test(&mut network, &data);
    Ok(())
}
