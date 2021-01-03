#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int_fast8_t
index_of(char const* word,
         char* const* const wordv,
         int_fast8_t start,
         int_fast8_t end)
{
  for (int_fast8_t i = start; i < end; i++) {
    if (strcmp(word, wordv[i]) == 0) {
      return i;
    }
  }

  return -1;
}

void
print_array(float const* arr, int_fast8_t r, int_fast8_t c)
{
  printf("---\n");
  printf("%" PRIdFAST8 " rows, %" PRIdFAST8 " cols\n", r, c);
  for (int_fast8_t y = 0; y < r; y++) {
    for (int_fast8_t x = 0; x < c; x++) {
      printf("[%2d] %.2f ", y * c + x, arr[y * c + x]);
    }
    printf("\n");
  }
  printf("---\n");
}

/**
 * attention_in is a token_count x token_count array.
 * attention_out is an empty word_count x word_count array
 */
void
merge(float* attention_in,
      float* attention_out,
      char* const* const tokenv,
      int_fast8_t token_count,
      char* const* const wordendsv,
      int_fast8_t word_count,
      int_fast8_t verbosity)
{
  if (verbosity > 0)
    printf("token_count: %d\n", token_count);

  if (verbosity > 1)
    print_array(attention_in, token_count, token_count);

  // merged_attention is word_count x token_count
  float* merged_attention;
  merged_attention = malloc(word_count * token_count * sizeof(float));

  if (verbosity > 1)
    print_array(merged_attention, token_count, word_count);

  // Step 1: merge attention *to* split words
  for (int_fast8_t token_from = 0; token_from < token_count; token_from++) {
    float attention_sum = 0;
    int_fast8_t last_seen_word_end = -1;

    for (int token_to = 0; token_to < token_count; token_to++) {
      attention_sum += attention_in[token_from * token_count + token_to];

      if (verbosity > 2)
        printf(
          "%d -> %.2f\n", token_from * token_count + token_to, attention_sum);

      // Is tokenv[token_to] a word end? Check from last seen word_end to end of
      // words. This could be a problem is it's not a word end in one word, but
      // is a word end in another.
      int_fast8_t possible_word_index = index_of(
        tokenv[token_to], wordendsv, last_seen_word_end + 1, word_count);
      if (possible_word_index >= 0) {
        // If it is a word end, assign last_seen_word_end and update
        // merged_attention
        last_seen_word_end = possible_word_index;

        if (verbosity > 2)
          printf("%d (merged) => %.2f\n",
                 token_from * word_count + last_seen_word_end,
                 attention_sum);

        merged_attention[token_from * word_count + last_seen_word_end] =
          attention_sum;
        attention_sum = 0;
      }
    }
  }

  if (verbosity > 1)
    print_array(merged_attention, token_count, word_count);

  // Step 2: merge attention *from* split words
  for (int_fast8_t word_j = 0; word_j < word_count; word_j++) {
    int_fast8_t word_i = -1;
    float attention_to_word = 0;
    int_fast8_t tokens_to_word_count = 0;

    for (int_fast8_t token_i = 0; token_i < token_count; token_i++) {
      attention_to_word += merged_attention[token_i * word_count + word_j];

      if (verbosity > 2)
        printf(
          "%d -> %.2f\n", token_i * word_count + word_j, attention_to_word);

      tokens_to_word_count++;

      int_fast8_t possible_word_i =
        index_of(tokenv[token_i], wordendsv, word_i + 1, word_count);
      if (possible_word_i >= 0) {
        word_i = possible_word_i;
        attention_out[word_i * word_count + word_j] =
          attention_to_word / tokens_to_word_count;

        if (verbosity > 2)
          printf("%d (merged) => %.2f\n",
                 word_i * word_count + word_j,
                 attention_to_word / tokens_to_word_count);

        attention_to_word = 0;
        tokens_to_word_count = 0;
      }
    }
  }

  free(merged_attention);
  if (verbosity > 1)
    print_array(attention_out, word_count, word_count);
}
