/* Neural Network Simulation
 *
 * Tyler Mau
 *
 * Network with 2 Inputs, 3 Hidden Nodes, and 2 Outputs
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define ALPHA 0.005
#define MAX_REF_VAL 100
#define MAX_INPUT 100
#define MAX_OUTPUT 100
 
#define HIDDEN_NODES 3
#define INPUT_NUM 2
#define OUTPUT_NUM 2
#define ITERATIONS 100000

#define TEST1 2
#define TEST2 4

/* Struct for Hidden Nodes */
typedef struct hidden_node {
   float data[INPUT_NUM];
   float weights[INPUT_NUM];
   float output;
   float bias;
}hidden_node;

/* Struct for Output Nodes */
typedef struct output_node {
   float weights[HIDDEN_NODES];
   float old_weights[HIDDEN_NODES];
   float output;
   float bias;
   float gradient;
}output_node;

/* Struct for Output Values (NOT DYNAMIC) */
typedef struct output_vals {
   float val1;
   float val2;
}output_vals;

/* Prints Node Data for All Nodes */
void print_nodes(hidden_node *h_nodes, output_node *o_nodes) {
   printf("\n         Hid_0   Hid_1   Hid_2   Out_0   Out_1\n");
   printf("Weight0: %.3f   %.3f   %.3f   %.3f   %.3f\n",
          h_nodes[0].weights[0], h_nodes[1].weights[0], h_nodes[2].weights[0],
          o_nodes[0].weights[0], o_nodes[1].weights[0]);

   printf("Weight1: %.3f   %.3f   %.3f   %.3f   %.3f\n",
          h_nodes[0].weights[1], h_nodes[1].weights[1], h_nodes[2].weights[1],
          o_nodes[0].weights[1], o_nodes[1].weights[1]);

   printf("Weight2:                         %.3f   %.3f\n",
          o_nodes[0].weights[2], o_nodes[1].weights[2]);

   printf("Biases:  %.3f   %.3f   %.3f   %.3f   %.3f\n",
          h_nodes[0].bias, h_nodes[1].bias, h_nodes[2].bias,
          o_nodes[0].bias, o_nodes[1].bias);

   printf("Output:  %.3f   %.3f   %.3f   %.3f   %.3f\n",
          h_nodes[0].output, h_nodes[1].output, h_nodes[2].output,
          o_nodes[0].output, o_nodes[1].output);

   printf("\n");
}

/* Initializes Neural Network */
void init_neural_network(hidden_node *h_nodes, output_node *o_nodes) {
   int count1 = 0, count2 = 0;

   /* Initialize Hidden Nodes */
   for (count1 = 0; count1 < HIDDEN_NODES; count1++) {
      for (count2 = 0; count2 < INPUT_NUM; count2++) {
         h_nodes[count1].data[count2] = 0;
         h_nodes[count1].weights[count2] = 0;//(float)(rand()) / RAND_MAX;
      }
      h_nodes[count1].bias = 0;//(float)rand() / RAND_MAX;
      h_nodes[count1].output = 0;
   }

   /* Initialize Output Nodes */
   for (count1 = 0; count1 < OUTPUT_NUM; count1++) {
      for (count2 = 0; count2 < HIDDEN_NODES; count2++) {
         o_nodes[count1].weights[count2] = 0;//(float)rand() / RAND_MAX;
         o_nodes[count1].old_weights[count2] = o_nodes[count1].weights[count2];
      }
      o_nodes[count1].bias = 0;//(float)rand() / RAND_MAX;
      o_nodes[count1].output = 0;
      o_nodes[count1].gradient = 0;
   }
}

/* Activation Function to Scale Data to 0 -> 1 */
float activation_function(float input) {
   return (1 / (1 + exp(input * -1)));
}

/* Calculates Hidden Node Outputs */
void calculate_hidden_nodes(hidden_node *h_nodes, float *inputs) {
   int count1 = 0, count2 = 0;
   float output = 0;

   for (count1 = 0; count1 < HIDDEN_NODES; count1++) {
      for (count2 = 0; count2 < INPUT_NUM; count2++) {
         h_nodes[count1].data[count2] = inputs[count2];
         output += inputs[count2] * h_nodes[count1].weights[count2];
      }
      output -= h_nodes[count1].bias;
      h_nodes[count1].output = activation_function(output);
      output = 0;
   }
}

/* Calculates Output Node Outputs */
void calculate_output_nodes(hidden_node *h_nodes, output_node *o_nodes) {
   int count1 = 0, count2 = 0;
   float output = 0;

   for (count1 = 0; count1 < OUTPUT_NUM; count1++) {
      for (count2 = 0; count2 < HIDDEN_NODES; count2++) {
         output += h_nodes[count2].output * o_nodes[count1].weights[count2];
      }
      output -= h_nodes[count1].bias;
      o_nodes[count1].output = activation_function(output);
      output = 0;
   }
}

output_vals *compute_neural_network(hidden_node *h_nodes, output_node *o_nodes, float *inputs) {
   output_vals *out_vals = calloc(1, sizeof(output_vals));

   calculate_hidden_nodes(h_nodes, inputs);
   calculate_output_nodes(h_nodes, o_nodes);
   
   /* Populates Struct for Returning Output Data (NOT DYNAMIC) */
   out_vals->val1 = o_nodes[0].output;
   out_vals->val2 = o_nodes[1].output;

   return out_vals;
}

void adjust_output_nodes(hidden_node *h_nodes, output_node *o_nodes, float *errors) {
   int count1 = 0, count2 = 0;
   float adj, gradient, output, bias_adj;

   for (count1 = 0; count1 < OUTPUT_NUM; count1++) {
      output = o_nodes[count1].output;
      gradient = output * (1 - output) * errors[count1];
      o_nodes[count1].gradient = gradient;

      for (count2 = 0; count2 < HIDDEN_NODES; count2++) {
         adj = ALPHA * h_nodes[count2].output * gradient;

         o_nodes[count1].old_weights[count2] = o_nodes[count1].weights[count2];
         o_nodes[count1].weights[count2] += adj;
      }
      bias_adj = ALPHA * 1 * gradient;
      o_nodes[count1].bias -= bias_adj;
   }
}

void adjust_hidden_nodes(hidden_node *h_nodes, output_node *o_nodes) {
   int count1 = 0, count2 = 0, count3 = 0;
   float adj, gradient, output, bias_adj, out_gradient = 0;

   for (count1 = 0; count1 < HIDDEN_NODES; count1++) {
      output = h_nodes[count1].output;

      for (count2 = 0; count2 < OUTPUT_NUM; count2++) {
         out_gradient += o_nodes[count2].gradient * o_nodes[count2].old_weights[count1];
      }

      gradient = output * (1 - output) * out_gradient;
      out_gradient = 0;

      for (count3 = 0; count3 < INPUT_NUM; count3++) {
         adj = ALPHA * h_nodes[count1].data[count3] * gradient;
         h_nodes[count1].weights[count3] += adj;
      }
      bias_adj = ALPHA * 1 * gradient;
      h_nodes[count1].bias -= bias_adj;
   }
}

output_vals *reference(int input1, int input2) {
   output_vals *output = calloc(1, sizeof(output_vals));

   output->val1 = input1 * TEST1;
   output->val2 = input2 * TEST2;

   return output;
}

void train_neural_network(hidden_node *h_nodes, output_node *o_nodes, float *outputs, int *inputs) {
   int count = 0;
   float errors[OUTPUT_NUM], expected[OUTPUT_NUM];
   output_vals *ref;

   ref = reference(inputs[0], inputs[1]);

   expected[0] = ref->val1 / MAX_OUTPUT;
   expected[1] = ref->val2 / MAX_OUTPUT;

   free(ref);

   /* Compute Output Errors */
   for (count = 0; count < OUTPUT_NUM; count++) {
      errors[count] = expected[count] - outputs[count];
   }

   adjust_output_nodes(h_nodes, o_nodes, errors);
   adjust_hidden_nodes(h_nodes, o_nodes);
}

void print_results(int *inputs, int *outputs) {
   printf("Inputs:     %d   %d\n", inputs[0], inputs[1]);
   printf("Outputs:    %d   %d\n", outputs[0], outputs[1]);
   printf("Expected:   %d   %d\n", inputs[0] * TEST1, inputs[1] * TEST2);
   printf("Error: %d   %d\n\n", outputs[0] - (inputs[0] * TEST1), outputs[1] - (inputs[1] * TEST2));
}

void scale_inputs(int *inputs, float *scaled_inputs) {
   int count = 0;
   for (count = 0; count < INPUT_NUM; count++) {
      scaled_inputs[count] = (float)inputs[count] / MAX_INPUT;
   }
}

int main(void) {
   int inputs[INPUT_NUM], scaled_outputs[OUTPUT_NUM], count = 0;
   float outputs[OUTPUT_NUM];
   float scaled_inputs[INPUT_NUM];
   hidden_node h_nodes[HIDDEN_NODES];
   output_node o_nodes[OUTPUT_NUM];
   output_vals *out_vals;

   /* Seed Random */
   srand(10);

   inputs[0] = 0;
   inputs[1] = 0;

   /* Scale Input Values to 0 -> 1 */
   scale_inputs(inputs, scaled_inputs);

   /* Initialize Neural Network */
   init_neural_network(h_nodes, o_nodes);
   print_nodes(h_nodes, o_nodes);

   for (count = 0; count < 0; count++) {
      /* Compute Neural Network */
      out_vals = compute_neural_network(h_nodes, o_nodes, scaled_inputs);

      //printf("INPUTS: %d   %d\n", (int)(h_nodes[0].data[0] * MAX_INPUT), (int)(h_nodes[0].data[1] * MAX_INPUT));

      outputs[0] = out_vals->val1;
      outputs[1] = out_vals->val2;

      /* Scale Outputs */
      scaled_outputs[0] = out_vals->val1 * MAX_OUTPUT;
      scaled_outputs[1] = out_vals->val2 * MAX_OUTPUT;

      /* Train Neural Network */
      train_neural_network(h_nodes, o_nodes, outputs, inputs);
      free(out_vals);

      inputs[0] = rand() % 10;
      inputs[1] = rand() % 10;
      scale_inputs(inputs, scaled_inputs);
   }

   //print_nodes(h_nodes, o_nodes);

   inputs[0] = 0;
   inputs[1] = 0;

   /* Test Computation */
   for (count = 0; count < 100000; count++) {
      inputs[0] = 5;
      inputs[1] = 5;
      scale_inputs(inputs, scaled_inputs);
      out_vals = compute_neural_network(h_nodes, o_nodes, scaled_inputs);

      outputs[0] = out_vals->val1;
      outputs[1] = out_vals->val2;

      scaled_outputs[0] = out_vals->val1 * MAX_OUTPUT;
      scaled_outputs[1] = out_vals->val2 * MAX_OUTPUT;

      print_results(inputs, scaled_outputs);
      //print_nodes(h_nodes, o_nodes);

      train_neural_network(h_nodes, o_nodes, outputs, inputs);
      //print_nodes(h_nodes, o_nodes);

   }

  return 0;
}
