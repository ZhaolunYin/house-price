use std::error::Error;
use std::io::{self, Write};

struct Data {
    input: Vec<f64>,
    output: f64,
}

fn predict(weights: &Vec<f64>, data: &Data) -> f64 {
    weights[0] * data.input[0] + 
        weights[1] * data.input[1] +
        weights[2] * data.input[2] +
        weights[3] * data.input[3] +
        weights[4] * data.input[4] +
        weights[5] * data.input[5] +
        weights[6] * data.input[6] +
        weights[7] * data.input[7] +
        weights[8] * data.input[8] +
        weights[9] * data.input[9] +
        weights[10] * data.input[10] +
        weights[11]
}

// Mean absolute error
fn loss(weights: &Vec<f64>, data: &Vec<Data>) -> f64 {
    let mut sum = 0.0;
    for d in data {
        // 3x + 1
        let prediction = predict(weights, d);
        let error = prediction - d.output;
        sum += error.abs();
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

fn yes_no(s: String) -> f64 {
    match s.as_str() {
        "yes" => 1.0,
        "no" => 0.0,
        _ => panic!("invalid")
    }
}


fn test(weights: &Vec<f64>) {
    println!("Enter house information:");

    let mut input = Vec::new();

    let names = [
        "area (sq. ft)",
        "bedrooms",
        "bathrooms",
        "stories",
        "mainroad (1=yes, 0=no)",
        "guestroom (1=yes, 0=no)",
        "basement (1=yes, 0=no)",
        "hotwaterheating (1=yes, 0=no)",
        "airconditioning (1=yes, 0=no)",
        "parking",
        "prefarea (1=yes, 0=no)",
    ];

    for name in names {
        print!("{name}: ");
        io::stdout().flush().unwrap();

        let mut value = String::new();
        io::stdin().read_line(&mut value).unwrap();

        input.push(value.trim().parse::<f64>().unwrap());
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
    let rdr = csv::Reader::from_path("Housing.csv");
    for result in rdr?.records() {
        let result = result?;
        data.push(
            Data {
                input: vec![
                    result[1].parse()?,
                    result[2].parse()?,
                    result[3].parse()?,
                    result[4].parse()?,
                    yes_no(result[5].to_string()),
                    yes_no(result[6].to_string()),
                    yes_no(result[7].to_string()),
                    yes_no(result[8].to_string()),
                    yes_no(result[9].to_string()),
                    result[10].parse()?,
                    yes_no(result[11].to_string()),
                ],
                output: result[0].parse()?,
            }
        )
    }
    let weights_filename = "weights.bin";
    let mut weights = vec![0.0; 12];

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

    let learning_rate = 0.00000001;

    let mut prev_loss = loss(&weights, &data);
    for i in 0..100000 {
        train(&mut weights, &data, learning_rate);

        if i % 1000 == 0 {
            let loss = loss(&weights, &data);
            println!("iteration {i:6}, loss = {}, improvement = {}", loss, prev_loss - loss);
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
    test(&weights);
    Ok(())
}
