from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from torch import nn, from_numpy
import torch
from random import randint, shuffle
import torchvision.datasets as dset
from sklearn.metrics import classification_report
import os

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import aiofiles
from lime import lime_image
from PIL import Image
from torchvision import transforms
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

class ClassifierModel(nn.Module):
    """
    A class that performs all neural network functionalities

    ...

    Attributes
    ----------
    image_feature_extractor : nn.Module object
        model that takes image as input and outputs 1000 dimension feature vector
    image_feature_size : int
        the size of image feature vector
    text_feature_extractor : nn.Module object
        model that takes sentence as input and outputs 768 dimension feature vector
    text_feature_size : int
        the size of text feature vector
    hidden_size : int
        number of neurons in the hidden layer(s)
    hidden_size : int
        number of neurons in the hidden layer(s)
    

    Methods
    -------
    forward(images, text)
        Performs forwards pass through the neural network
    
    """

    def __init__(self):
        """
        Performs initializations of all needed variables
        
        Parameters
        ----------
        None

        Returns
        ----------
        None

        """

        super(ClassifierModel, self).__init__()

        # Incorporating pre-trained models
        image_feature_extractor = models.resnet18(pretrained=True)
        self.image_feature_extractor = torch.nn.Sequential(*list(image_feature_extractor.children())[:-1])
        self.image_feature_extractor.eval()
        self.text_feature_extractor = SentenceTransformer('stsb-roberta-base')
        self.text_feature_extractor.eval()
        self.image_feature_size = 512
        self.text_feature_size = 768
        self.seperator_size = 0
        self.out_size = 1
        self.hidden_size = 1024
        
        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.image_feature_size + self.text_feature_size + self.seperator_size, out_features=self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.out_size, bias=False),
            nn.Sigmoid()
        )


    def forward(self,images,text):
        """Performs forwards pass through the neural network

        Both images and text is of a particular `batch size` during training, and of
        size 1 during predictions

        Parameters
        ----------
        images : tensor
            Represents a collection of images that must be fed to our model. It must be of 
            size `(batch_size,image_width,image_height,no_of_channels)`
        
        text : list
            Represents a list of sentences(representing image captions) that must be fed to our model. It must be of 
            size `(batch_size,)`

        Returns
        ------
        images : tensor
            Returns the output of the model after `images` and `text` are used as input
        
        """

        # Extracting features for image and text
        with torch.no_grad():
            image_feature_vecs = self.image_feature_extractor.forward(images).flatten(start_dim=1)
            text_feature_vecs = from_numpy(self.text_feature_extractor.encode(text)).cuda()
        #seperator_vecs = torch.zeros(images.shape[0],self.seperator_size).cuda()
        
        # Concating features and obtaining final outputs
        fc_input = torch.cat((image_feature_vecs,text_feature_vecs),dim=-1)
        # fc_input = torch.cat((image_feature_vecs,seperator_vecs,text_feature_vecs),dim=-1)
        output = self.fc(fc_input)

        return output

class DatasetHandler(dset.CocoCaptions): 
    """ A class that performs all dataset functionalities

    ...
    
    Attributes
    ----------
    last_index : int
        keeps track of the index till which batch size was computed
    length : int
        number of training samples
    imageTransformer : transforms object
        pipeline that takes an image of type `PIL object`, resizes it to  size (224,224) 
        and converts it into tensor 
    negative_samples : int
        number of negative samples per positive sample
    
    
    Methods
    -------
    get_next_batch(batch_size)
        generates next batch of input of size `(negative_samples + 1) * batch_size * 5`
    get_all()
        gives all the samples from the dataset regardless of `last_index`
    _generate_combinations(dataset)
        generates negative samples and performs shuffling on current batch
    transform_Image(image)
        applies `imageTransformer` on a single image
    
    """
    
    def __init__(self,
                root,
                annFile,
                negative_samples = 1):
        """Performs initializations of all needed variables.
    
        Parameters
        ----------
        root : str
            Path to the directory containing the dataset comprising 
            of images
        annFile : str
            Path to the json file containing mapping of image and 
            its annotations
        negative_samples : int, optional
            number of negative sample(s) per positive sample.Each batch will comprise of 
            positive and negative samples in the ratio `1:negative_samples` 
    
    
        Returns
        ------
        None
    
        """
    
        super(dset.CocoCaptions,self).__init__(root,annFile)
        self.last_index = 0
        self.length = len(self.ids)
        self.imageTransformer = transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                    ])
        self.negative_samples = negative_samples
    
    def get_next_batch(self,batch_size):
        """generates next batch of input of size `(negative_samples + 1) * batch_size * 5`.
    
        Parameters
        ----------
        batch_size : int
            number of positive samples in the batch
    
    
        Returns
        ------
        dataset : tuple
            tuple of size 2 where `dataset[0]` contains the input to model and 
            `dataset[1]` contains the label for the input.
    
            Further `dataset[0][0]` contains images and `dataset[0][1]` contains text
        dataset_empty: bool
            indicates whether end of the dataset is reached
    
        """
        dataset = []
        dataset_empty = False
        for index in range(0,batch_size):
            curr_index = self.last_index + index
            if curr_index >=self.length:
                dataset_empty = True
                break
            dataset.append(self.__getitem__(curr_index))
        if dataset_empty:
            self.last_index = 0
        else:
            self.last_index += batch_size
        dataset = self._generate_combinations(dataset)
    
        return dataset, dataset_empty
    
    def get_all(self):
        """gives all the samples from the dataset regardless of `last_index`.
        The batch would be of size `(negative_samples + 1) * length * 5`.
    
        Parameters
        ----------
        None
    
    
        Returns
        ------
        data : tuple
            tuple of size 2 where `data[0]` contains the input to model and 
            `data[1]` contains the label for the input
    
            Further `data[0][0]` contains images and `data[0][1]` contains text
    
        """
        last_index = self.last_index
        self.last_index = 0
        res = self.get_next_batch(64)
        self.last_index = last_index
    
        return res
    
    def _generate_combinations(self,dataset):
        """generates negative samples and performs shuffling on current batch.
        The batch size of the final dataset will be `(negative_samples + 1)* batch_size * 5`
    
        Parameters
        ----------
        dataset: list
            array of ids that represents the positive samples within the current batch
    
    
        Returns
        ------
        dataset_X : tuple
            tuple where `dataset_X[0]` contains images and `dataset_X[1]` contains text
        dataset_Y : tensor
            contains the labels for the samples in `dataset_X`
    
        """
    
        dataset_X_images = []
        dataset_X_texts = []
        dataset_Y = []
        for index in range(0,len(dataset)):
            curr_data = dataset[index]
            for i in range(0,len(curr_data[1])):
                # Data with label 'yes'
                dataset_X_images.append(self.transform_Image(curr_data[0]) )
                dataset_X_texts.append(curr_data[1][i])
                dataset_Y.append(1)
    
                # Data with label 'no'
                for _ in range(0,self.negative_samples):
                    ind = index
                    while ind == index:
                        ind = randint(0,len(dataset)-1)
    
                    text_ind = randint(0,len(dataset[ind][1])-1)
                    dataset_X_images.append(self.transform_Image(curr_data[0]))
                    dataset_X_texts.append(dataset[ind][1][text_ind])
                    dataset_Y.append(0)
    
        # Shuffling the batch
        batch = list(zip(dataset_X_images, dataset_X_texts, dataset_Y ))
        shuffle(batch)
        dataset_X_images, dataset_X_texts, dataset_Y = zip(*batch)
    
        dataset_X = ( torch.stack(dataset_X_images).cuda(), dataset_X_texts )
        dataset_Y = torch.Tensor(dataset_Y).cuda()
    
        return (dataset_X,dataset_Y)
    
    def transform_Image(self,image):
        """applies `imageTransformer` on a single image
    
        Parameters
        ----------
        image: PIL object
            a single image of size `image_width * image_height * no_of_channels`
    
    
        Returns
        ------
        img : tensor
            tensor of size `( 224 * 224 * no_of_channels )` that represents the image
    
        """
    
        img = self.imageTransformer(image)
    
        return img

class ZeroshotClassifier():
    """
    A class that performs model training, utilizing the dataset.

    ...

    Attributes
    ----------
    model : ClassifierModel object
        class that has all model operations predefined 
    train_dataset : DatasetHandler object
        class that has all training dataset operations predefined 
    val_dataset : DatasetHandler object
        class that has all validation dataset operations predefined 
    batch_size : int, optional
        number that represents number of positive samples in one batch 
    learning_rate : int, optional
        `learning rate` for the optimizer
    validate_batch: int, optional
        during each epoch, number of batches after which validation dataset to
        be passed through neural network
    display_report: bool, optional
        flag indicating whether detailed `sklearn classification_report` must
        be displayed for each validation run
    loss_func: nn.loss object
        the loss function to be used during training
    optimizer": torch.optim object
        optimizer that will updates weights of the model using `loss_func` value 

    Methods
    -------
    train(epochs)
        performs training of the model for `epochs` number of epochs
    predict(image,text)
        retrives prediction for a single pair of `( image , text )`
    validate(batch,epoch,loss)
        performs validation of the model and reports validation accuracy
    save_model(path)
        saves the learned model at a particular `path`
    load_model(path)
        loads the learned model from a particular `path`
    _evaluation_metrics(pred,truth)
        generates the classification_report using predicition and truth values
    _calculate_accuracy(pred,truth)
        calculates the accuracy achieved using prediction and truth values

    """
    def __init__(self,
                 train_data_dir,
                 train_annotation_path,
                 val_data_dir,
                 val_annotation_path,
                 batch_size=32,
                 learning_rate=1e-4,
                 validate_batch=None,
                 display_report=False):
        """Performs initializations of all needed variables.

        Parameters
        ----------
        train_data_dir : str
            Path to the directory containing the training images
        train_annotation_path : str
            Path to the json file containing mapping of training image and 
            its annotations
        val_data_dir : str
            Path to the directory containing the validation images
        val_annotation_path : str
            Path to the json file containing mapping of validation image and 
            its annotations
        batch_size : int, optional
            number that represents number of positive samples in one batch 
        learning_rate : int, optional
            `learning rate` for the optimizer
        validate_batch: int, optional
            during each epoch, number of batches after which validation dataset to
            be passed through neural network
        display_report: bool, optional
            flag indicating whether detailed `sklearn classification_report` must
            be displayed for each validation run


        Returns
        ------
        None

        """
        self.model = ClassifierModel()
        self.model.cuda()
        self.train_dataset = DatasetHandler(root=train_data_dir,annFile=train_annotation_path)
        self.val_dataset = DatasetHandler(root=val_data_dir,annFile=val_annotation_path)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validate_batch = validate_batch
        self.display_report = display_report
        self.loss_func = nn.BCELoss()

        params = [param for param in self.model.fc.parameters()]
        self.optimizer = torch.optim.Adam(params,lr=learning_rate)

    def train(self,epochs):
        """performs training of the model for `epochs` number of epochs

        Parameters
        ----------
        epochs : int
            represents the number of epochs the model must be trained for


        Returns
        ------
        None

        """
        for epoch in range(1,epochs+1):
            dataset_end = False
            total_loss = 0
            batch = 1
            while not dataset_end:
                (X,y), dataset_end = self.train_dataset.get_next_batch(self.batch_size)

                pred = self.model.forward(X[0],X[1])
                loss = self.loss_func(pred.view(-1),y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.cpu().detach().numpy()
                if dataset_end or ( self.validate_batch and (batch % self.validate_batch == 0) ):
                    self.validate(epoch, batch, total_loss/self.validate_batch)
                    total_loss = 0
                batch += 1
        
        return

    def predict(self,image,text):
        """retrives prediction for a single pair of `( image , text )`

        Parameters
        ----------
        image : PIL object
            image that will be fed through the model for prediction
        text : str
            text that will be fed through the model for prediction
            

        Returns
        ------
        pred: tensor
            single value between [0,1] representing the likeliness of
            text representing the image

        """

        img = self.train_dataset.transform_Image(image)
        img = torch.unsqueeze(img, 0)
        text = [text]
        self.model.eval()
        pred = self.model.forward(img,text)

        self.model.train()

        return pred
    
    def validate(self, epoch, batch, loss):
        """performs validation of the model and reports validation accuracy

        Parameters
        ----------
        epoch : int
            current epoch number
        epoch : int
            current batch number for the `epoch`
        loss : int
            loss incurred since last validation run
            

        Returns
        ------
        None

        """
        self.model.eval()
        dataset_end = False
        batch_size = 64
        with torch.no_grad():
            (X,y), dataset_end = self.val_dataset.get_next_batch(batch_size)
            pred = self.model.forward(X[0],X[1])
        if dataset_end:
            self.val_dataset.last_index = 0
        self.model.train()
        accuracy, report = self._evaluation_metrics(pred.view(-1).cpu(),y.cpu())
        if self.display_report:
            print("Epoch: {} \t Batch: {} \t Loss: {} \t Accuracy Achieved: {} \n {}".format(epoch, batch, loss, accuracy, report))
        else:
            print("Epoch: {} \t Batch: {} \t Loss: {} \t Accuracy Achieved: {} \n".format(epoch, batch, loss, accuracy))

        return
    
    def save_model(self,path):
        """saves the learned model at a particular `path`

        Note: If `path` does not exists the directory is manually created

        Parameters
        ----------
        path : str
            the path where the model parameters must be saved
            

        Returns
        ------
        None

        """
        if not os.path.exists(os.path.dirname(path)):
            print("[Warning]: The path to save the model does not exists. Folder manually Created")
            os.mkdir(os.path.dirname(path))
        
        torch.save(self.model.state_dict(),path)
    
    def load_model(self,path):
        """loads the learned model from a particular `path`

        Note: If `path` does not exists error is displayed, but the model
              will continue to be trained.

        Parameters
        ----------
        path : str
            the path where the model parameters is saved
            

        Returns
        ------
        None

        """
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.train()
        else:
            print("[Error]: The path to saved model does not exists")

    def _evaluation_metrics(self,pred,truth):
        """generates the classification_report using predicition and truth values

        Parameters
        ----------
        pred : tensor
            array containing predictions provided by the model for a specific
            list of examples
        truth : tensor
            array containing actual labels for the same list of examples
            

        Returns
        ------
        accuracy: int
            value between [0,100] that represents the accuracy achieved
        report: str
            classification report generated by sklearn's `classification_report`

        """
        pred = torch.round(pred).detach()

        accuracy = self._calculate_accuracy(pred,truth)
        report = classification_report(truth,pred.detach().numpy())

        return (accuracy,report)
    
    def _calculate_accuracy(self,pred,truth):
        """calculates the accuracy achieved using prediction and truth values

        Parameters
        ----------
        pred : tensor
            array containing predictions provided by the model for a specific
            list of examples
        truth : tensor
            array containing actual labels for the same list of examples
            

        Returns
        ------
        accuracy: int
            value between [0,100] that represents the accuracy achieved

        """

        return (pred == truth).sum() / pred.shape[0] * 100.0

if __name__ == "__main__":
    class InputText(BaseModel):
        sentence: str
        class Config:
            schema_extra = {
                "example": {
                    "sentence": "Bus"
                }
            }

    app = FastAPI()

    def get_prediction():
        global model,image_transformer,text
        sentences = [text]
        image = Image.open("input.jpg").convert('RGB')
        img = image_transformer(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        model.eval()
        with torch.no_grad():
            pred = model.forward(img,sentences)
        return pred.sum()

    def predict_image(images):
        global model,image_transformer,text
        # images = torch.from_numpy(image).float()
        pil_images = []
        for image in images:
            pil_images.append(Image.fromarray(image.astype('uint8'), 'RGB'))
        transformed_images = []
        for image in pil_images:
            transformed_images.append(image_transformer(image))
        images = torch.stack(transformed_images)
        images = images.cuda()
        model.eval()
        with torch.no_grad():
            pred = model.forward(images,[text]*images.shape[0])
        pred = pred.cpu()
        output = torch.cat((pred*-1 + 1,pred),dim=-1)
        return output.numpy()

    def explain_image():
        global explainer

        image=Image.open('input.jpg').convert('RGB')
        image = np.array(image)
        print('working')
        explanation = explainer.explain_instance(image, predict_image, top_labels=2, hide_color=0, num_samples=2000)

        temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=10, hide_rest=False)

        plt.imsave('output.jpg',mark_boundaries(temp, mask))


    @app.on_event("startup")
    def load_model():
        global model, explainer, image_transformer, image_resizer

        # Loading trained classifier model
        model = ClassifierModel()
        model.cuda()
        model.load_state_dict(torch.load('model/weights.pt'))


        # Initializing lime explainer for image
        explainer = lime_image.LimeImageExplainer()

        # Initializing transforms needed for image input

        image_transformer = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                        ])
        image_resizer = transforms.Resize((224,224))


    @app.get("/get_input_image/")
    async def get_image():
        return FileResponse("input.jpg")

    @app.get("/get_output_image/")
    async def get_image():
        return FileResponse("output.jpg")



    @app.post("/predict_image/")
    async def make_inference(sentence: str,file: UploadFile = File(...)):
        global text
        async with aiofiles.open('input.jpg', 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)  # async write
        text = sentence
        prob = float(get_prediction())
        winner = 1
        if prob < 0.5:
            winner = 0

        explain_image()
        return {"winner":1,'positiveRating':prob,'negativeRating':1-prob}

    from colabcode import ColabCode
    server = ColabCode(port=10000, code=False)

    server.run_app(app=app)
