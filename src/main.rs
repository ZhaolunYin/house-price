use std::error::Error;
use std::io::{self, Write};

struct Data {
    input: Vec<f64>,
    output: f64,
}

fn predict(weights: &Vec<f64>, data: &Data) -> f64 {
    let mut sum = 0.0;
    for i in 0..(weights.len() - 1) {
        sum += weights[i] * data.input[i];
    }
    sum + weights.last().unwrap()
}

// Mean absolute error
fn loss(weights: &Vec<f64>, data: &Vec<Data>) -> f64 {
    let mut sum = 0.0;
    for d in data {
        // 3x + 1
        let prediction = predict(weights, d);
        let error = prediction - d.output;
        sum += error * error;
    }
    sum / (data.len() as f64)
}

fn train(weights: &mut Vec<f64>, data: &Vec<Data>, learning_rate: f64) {
    let mut gradients = vec![0.0; weights.len()];
    for d in data {
        let prediction = predict(weights, d);
        let miss = prediction - d.output;
        for i in 0..d.input.len() {
            gradients[i] += 2.0 * miss * d.input[i];
        }
        gradients[d.input.len()] += 2.0 * miss;
    }
    let n = data.len() as f64;
    for i in 0..weights.len() {
        gradients[i] /= n;
        weights[i] -= learning_rate * gradients[i];
    }
}

fn test(weights: &Vec<f64>, means: &Vec<f64>, std_ds: &Vec<f64>) {
    println!("Enter house information:");

    let mut input = Vec::new();

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

        input.push(value.trim().parse::<f64>().unwrap());
    }
    for i in 0..input.len() {
        input[i] = (input[i] - means[i]) / std_ds[i];
    }

    let data = Data {
        input,
        output: 0.0,
    };

    let prediction = predict(weights, &data);

    println!();
    println!("Predicted price: ${:.2}", prediction);
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut data = Vec::new();
    let rdr = csv::Reader::from_path("kc_house_data.csv");
    for result in rdr?.records() {
        let result = result?;
        data.push(
            Data {
                input: vec![
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
    // Normalize inputs:
    let n = data.len() as f64;
    let cols = data[0].input.len();
    let mut means = vec![0.0; cols];
    let mut std_ds = vec![0.0; cols];

    for d in &data {
        for i in 0..cols {
            means[i] += d.input[i];
        }
    }
    for i in 0..cols {
        means[i] /= n;
    }

    for d in &data {
        for i in 0..cols {
            let diff = d.input[i] - means[i];
            std_ds[i] += diff * diff;
        }
    }
    for i in 0..cols {
        std_ds[i] /= n;
        std_ds[i] = std_ds[i].sqrt();
    }

    for d in &mut data {
        for i in 0..cols {
            d.input[i] = (d.input[i] - means[i]) / std_ds[i];
        }
    }

    let weights_filename = "weights.bin";
    let mut weights = vec![0.0; 7];

    match std::fs::read(weights_filename) {
        Ok(bytes) => {
            if bytes.len() == weights.len() * 8 {
                for (i, chunk) in bytes.chunks_exact(8).enumerate() {
                    let arr: [u8; 8] = chunk.try_into()?;
                    weights[i] = f64::from_le_bytes(arr);
                }
                println!("loaded weights from {weights_filename}");
            } else {
                println!(
                    "{weights_filename} has the wrong size ({} bytes), starting from zeros",
                    bytes.len()
                );
            }
        }
        Err(_) => println!("no {weights_filename} found, starting from zeros"),
    }

    let learning_rate = 0.1;

    let mut prev_loss = loss(&weights, &data);
    for i in 0..1000 {
        train(&mut weights, &data, learning_rate);

        if i % 10 == 0 {
            let loss = loss(&weights, &data);
            println!("iteration {i:6}, loss = {:8}, improvement = {:8}", loss, prev_loss - loss);
            prev_loss = loss;
        }
    }
    let mut bytes = Vec::with_capacity(weights.len() * 8);
    for weight in &weights {
        bytes.extend_from_slice(&weight.to_le_bytes());
    }
    std::fs::write(weights_filename, bytes)?;
    print!("weights = ");
    for weight in &weights {
        print!("{weight} ");
    }
    println!();
    test(&weights, &means, &std_ds);
    Ok(())
}
