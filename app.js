const fitButton = document.getElementById('fit-btn');
const inputNumber = document.getElementById('input-number');
const predictButton = document.getElementById('predict-btn');
const result = document.getElementById("result");

let model;

async function entrenarModelo() {

  model = tf.sequential();

  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  });

  const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1])
  const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1])

  await model.fit(xs, ys, { epochs: 350 });


};

predictButton.addEventListener("click", () => {

  const inputValue = inputNumber.value;

  if (inputValue === "") {
    alert("Ingrese un número en el input");
    return;
  }

  const prediction = model.predict(tf.tensor2d([+inputValue], [1, 1]))

  prediction.array().then(value => {
    
    result.innerText = `El resultado de predecir ${inputValue} es: ${value[0][0]}`;
  })
})

fitButton.addEventListener("click", () => {
  entrenarModelo();

  alert("Entrenamiento del modelo finalizado")

  inputNumber.style.display = "inline-block";
  predictButton.style.display = "inline-block"
})