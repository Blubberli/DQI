from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from torch import nn


class RobertaMultiTask(RobertaForSequenceClassification):
    """
    A model for classification or regression with 4 different heads, each heads learns to predict
    a DQI dimension.
    The text features are processed with Roberta. The weights of Roberta are shared. For each task a small
    task-specific classifier is trained. The losses are summed together and backpropagated.

    This class expects a transformers.RobertaConfig object, and the config object
    needs to have two additional properties manually added to it:
      `text_feat_dim` - The length of the BERT vector.
      `numerical_feat_dim` - The number of numerical features.
    """

    def __init__(self, roberta_config_task1, roberta_config_task2, roberta_config_task3, roberta_config_task4):
        # Call the constructor for the huggingface `RobertaForSequenceClassification`
        # class, which will do all of the BERT-related setup. The resulting ROBERTA
        # model is stored in `self.roberta`. We can use any of the configs, as the Roberta-specific parameters do not change between tasks
        super().__init__(roberta_config_task1)

        # for each task we have a different number of labels
        self.num_labels_task1 = roberta_config_task1.num_labels
        self.num_labels_task2 = roberta_config_task2.num_labels
        self.num_labels_task3 = roberta_config_task3.num_labels
        self.num_labels_task4 = roberta_config_task4.num_labels

        self.config1 = roberta_config_task1
        self.config2 = roberta_config_task2
        self.config3 = roberta_config_task3
        self.config4 = roberta_config_task4

        # create task-specific heads
        self.classifier_task1 = RobertaClassificationHead(roberta_config_task1)
        self.classifier_task2 = RobertaClassificationHead(roberta_config_task2)
        self.classifier_task3 = RobertaClassificationHead(roberta_config_task3)
        self.classifier_task4 = RobertaClassificationHead(roberta_config_task4)

    def get_loss(self, logits, labels, config):
        loss = None
        # compute the loss as it is done in the original robertaforseqclassification code
        if labels is not None:
            if config.problem_type is None:
                if config.num_labels == 1:
                    config.problem_type = "regression"
                elif config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    config.problem_type = "single_label_classification"
                else:
                    config.problem_type = "multi_label_classification"

            if config.problem_type == "regression":
                loss_fct = MSELoss()
                if config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, config.num_labels), labels.view(-1))
            elif config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        return loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels_task1=None,
            labels_task2=None,
            labels_task3=None,
            labels_task4=None,
    ):
        r"""
        perform a forward pass of our model.

        This has the same inputs as `forward` in `RobertaForSequenceClassification`.
        It returns an aggregated loss over all task-specific losses
        """
        return_dict = return_dict if return_dict is not None else self.config1.use_return_dict
        # Run the text through the ROBERTA model. Invoking `self.roberta` returns
        # outputs from the encoding layers, and not from the final classifier.
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # take <s> token (equiv. to [CLS])
        sequence_output = outputs[0]
        # retrieve the logits for each task
        logits_task1 = self.classifier_task1(sequence_output)
        logits_task2 = self.classifier_task2(sequence_output)
        logits_task3 = self.classifier_task3(sequence_output)
        logits_task4 = self.classifier_task4(sequence_output)
        # create one list of logits
        all_logits = [logits_task1, logits_task2, logits_task3, logits_task4]
        all_labels = [labels_task1, labels_task2, labels_task3, labels_task4]
        all_losses = []
        configs = [self.config1, self.config2, self.config3, self.config4]
        for i in range(len(all_logits)):
            loss = self.get_loss(logits=all_logits[i], labels=all_labels[i], config=configs[i])
            all_losses.append(loss)
        loss = sum(all_losses)

        # return the result as it is done in the original code, loss, logits, hidden states and attentions
        return MultiSequenceClassifierOutput(
            loss=loss,
            logits=logits_task1,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            logits_task1=logits_task1,
            logits_task2=logits_task2,
            logits_task3=logits_task3,
            logits_task4=logits_task4
        )


@dataclass
class MultiSequenceClassifierOutput(SequenceClassifierOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        logits_task1:
        logits_task2:
        logits_task3:
        logits_task4:
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits_task1: torch.FloatTensor = None
    logits_task2: torch.FloatTensor = None
    logits_task3: torch.FloatTensor = None
    logits_task4: torch.FloatTensor = None
