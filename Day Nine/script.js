import {MnistData} from './data.js';

// Display examples from dataset
async function showExamples(data){
    const surface = tfvis.visor().surface({name : 'Input data examples',tab:'Input Data'});

    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    for(let i=0;i<numExamples;i++){
        const imageTensor = tf.tidy(() => {
            return examples.xs.slice([i,0],[1,examples.xs.shape[1]]).reshape([28,28,1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin:4px';
        await tf.browser.toPixels(imageTensor,canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

//This function is called once the page loads
async function run(){
    const data = new MnistData();
    await data.load();
    await showExamples(data);

    const model = getModel();
    tfvis.show.modelSummary({name:'Model Architecture'},model);

    await train(model,data);
    await showAccuray(model,data);
    await showConfusion(model, data);
}

// Code to define a CNN model
function getModel(){
    const model = tf.sequential();
    
    const i_w = 28;
    const i_h = 28;
    const i_c = 1;

    model.add(tf.layers.conv2d({
        inputShape : [i_w,i_h,i_c],
        kernelSize : 5,
        filters : 8,
        strides : 1,
        activation : 'relu',
        kernelInitializer : 'varianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({poolSize:[2,2], strides:[2,2]}));

    model.add(tf.layers.conv2d({
        inputShape : [i_w,i_h,i_c],
        kernelSize : 5,
        filters : 8,
        strides : 1,
        activation : 'relu',
        kernelInitializer : 'varianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({poolSize:[2,2], strides:[2,2]}));
    
    model.add(tf.layers.flatten());

    const num_op_classes = 10;
    model.add(tf.layers.dense({
        units : num_op_classes,
        kernelInitializer : 'varianceScaling',
        activation : 'softmax'
    }));

    const optimizer = tf.train.adam();
    model.compile({
        optimizer : optimizer,
        loss : 'categoricalCrossentropy',
        metrics:['accuracy']
    });

    return model;
}

// Train CNN-Model

async function train(model,data){
    const metrics = ['loss','val_loss','acc','val_acc'];
    const container = {
        name : 'Model Training', styles : {height : '1000px'}
    };

    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const batch_size = 512;
    const train_data_size = 5500;
    const test_data_size = 1000;

    const [trainX,trainY] = tf.tidy(()=> {
        const d = data.nextTrainBatch(train_data_size);
        return [
            d.xs.reshape([train_data_size,28,28,1]),
            d.labels
        ];
    });

    const [testX,testY] = tf.tidy(() => {
        const d = data.nextTestBatch(test_data_size);
        return [
        d.xs.reshape([test_data_size,28,28,1]),
        d.labels
    ];
    });

    return model.fit(trainX,trainY,{
        batchSize : batch_size,
        validationData : [testX,testY],
        epochs : 10,
        shuffle : true,
        callbacks : fitCallbacks
    })
}
const classNames = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine'];

function doJadoo(model,data,testDataSize = 500){
    const i_w = 28;
    const i_h = 28;
    const testData = data.nextTestBatch(testDaaSize);
    const testxs = testData.xs.reshape([testDataSize, i_w,i_h,1]);
    const labels = testData.labels.argMax([-1]);
    const preds = model.predict(testxs).agrgmax([-1]);

    texxs.dispose();
    return [preds, labels];
}

async function showAccuray(model, data){
    const [preds,labels] = doJadoo(model,data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels,preds);
    const container = {name : 'Accuracy', tab : 'Evaluation'};
    tfvis.show.perClassAccuracy(container,classAccuracy,classNames);

    labels.dispose();
}

async function showConfusion(model,data){
    const [preds,labels] = doJadoo(model,data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels,preds);
    const container = {name : 'Confusion Matrix', tab : 'Evaluation'}
    tfvis.render.confusionMatrix(
        container, {values:confusionMatrix}, classNames);

        labels.dispose();
    }


document.addEventListener('DOMContentLoaded',run);