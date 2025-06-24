import numpy as np


def estimate_success_rate_at_k_per_problem(n: int, c: int, k: int) -> float:
    """
    :param n: number of total attempts on this problem.
    :param c: number of correct attempts on this problem.
    :param k: k in pass_i@$k$.
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def calculate_pass_k_unbiased(true_labels, k_values, return_mean=True, max_num_answers=None):
    """
    Estimate the success rate (unbiased estimate) at k for each problem, and then take the mean.
    Returns a dictionary of k values to pass@k results.
    If max_num_answers is not None, then we will only consider the first max_num_answers attempts.

    Args:
        true_labels: (num_problems x num_answers)
        k_values: list of k values to compute pass@k for
        max_num_answers: int, maximum number of answers to consider
        return_mean: bool, whether to return the mean of the pass@k results, or per problem.
    """
    num_problems, total_num_answers = true_labels.shape
    pass_at_k_results = {}
    for k in k_values:
        if max_num_answers is not None:
            num_answers = min(max_num_answers, total_num_answers)
        else:
            num_answers = total_num_answers
        assert k <= num_answers
        c = np.sum(true_labels[:, :num_answers] == 1, axis=1)
        all_r = np.array([
            estimate_success_rate_at_k_per_problem(num_answers, c[p], k)
            for p in range(num_problems)
        ])
        if return_mean:
            pass_at_k_results[k] = np.nanmean(all_r)
        else:
            pass_at_k_results[k] = all_r
    return pass_at_k_results


def calculate_pass_k_gt(true_labels, k_values, return_mean=True):
    """
    Compute Pass@k for a matrix of shape (num_problems x num_answers),
    where each element is binary correctness (0 or 1).

    Args:
        true_labels: (num_problems x num_answers)
        k_values: list of k values to compute pass@k for
        return_mean: bool, whether to return the mean of the pass@k results, or per problem.

    Note: this assumes that the samples are iid. For an unbiased estimate, use pass_k_unbiased.
    """
    num_problems, num_answers = true_labels.shape
    pass_at_k_results = {}
    for k in k_values:
        if k <= num_answers:
            pass_k = np.any(true_labels[:, :k] == 1, axis=1)
            if return_mean:
                pass_at_k_results[k] = np.nanmean(pass_k)
            else:
                pass_at_k_results[k] = pass_k.astype(int)
    return pass_at_k_results


def calculate_majority_M_at_k(all_answers, y, k, topM=1, return_mean=False, return_idx=False,majority_select="majority"): 
    """
    TODO: check bias of majority vote

    Calculate if the correct answer is in the top M answers of the first k attempts.

    Args:
        all_answers = List[List] = list of answers per problem [num_problems x num_answers]
        y: array =  answer correctness[num_problems x num_answers]
        k: int = number of attempts to consider per problem
        topM: int = number of top answers to consider per problem.
        return_mean: bool = whether to return the mean of the results, or per problem.
        return_idx: bool = whether to return the index of the top-M answer.
        majority_select: str = "majority" or "one_sample"
    return:
        - all_problems: array of 0/1 indicating if the problem is correct
        - top_k_idx: index of the top-k answer
    """
    correct_problems = 0
    total_problems = 0

    num_problems = len(all_answers)
    all_problems = np.zeros(num_problems)
    all_topk_idx = np.zeros((num_problems, topM))*np.nan

    for problem_idx in range(num_problems):
        # Note here we add 1 to k to match the pass@k results
        # Look at the first k attempts: 
        answers = all_answers[problem_idx][:k]
        answers_correct = y[problem_idx][:k]    
        answer_counts, answer_correctness, answer_idx = {}, {}, {}
        for idx, (ans, is_correct) in enumerate(zip(answers, answers_correct)):
            if ans == 'NO_ANSWER':
                raise ValueError(f"NO_ANSWER for problem {problem_idx} at attempt {idx}")
                continue
            if ans not in answer_counts:
                answer_counts[ans] = 0
                answer_correctness[ans] = []
                answer_idx[ans] = []
            answer_counts[ans] += 1
            answer_correctness[ans].append(is_correct)
            answer_idx[ans].append(idx)

        if answer_counts:
            total_problems += 1
            top_k_answers = sorted(
                answer_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:topM]
            # Get the indices of the top-M answers
            top_k_idx = [answer_idx[ans] for ans, _ in top_k_answers]
        else:
            top_k_idx = []
        # topk_positive: number of queries where the top k answer is positive(correct)
        # prediction_accuracy: accuracy of the prediction (selected sample) for each query
        # Pick the first sample as the sample
        sample_idx = 0

        if majority_select == "majority":
            # List of lists of indices
            for top_id, (ans, _) in enumerate(top_k_answers):
                # If this answer is correct by majority
                if sum(answer_correctness[ans]) > len(answer_correctness[ans]) / 2:
                    correct_problems += 1
                    all_problems[problem_idx] = 1
                    all_topk_idx[problem_idx] = top_k_idx[top_id][sample_idx]
                    break 
        elif majority_select == "one_sample":
            for top_id, (ans, _) in enumerate(top_k_answers):
                sample_idx = 0
                correct_sample = answer_correctness[ans][sample_idx]
                if correct_sample:
                    correct_problems += 1
                    all_problems[problem_idx] = 1
                    all_topk_idx[problem_idx] = top_k_idx[top_id][sample_idx]
                    break

    if return_mean:
        output1 = correct_problems / total_problems
    else:
        output1 = all_problems

    if return_idx:
        output2 = all_topk_idx
    else:
        output2 = None

    return output1, output2
