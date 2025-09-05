from ._main import main, build_arg_parse
from .utils import (
    iter_quiz_files,
    write_jsonl,
    read_jsonl,
)
from .manager.quiz import (
    extract_topics,
    validate_mcq,
    generate_questions,
    select_questions,
    ai_generate_mcqs_for_topic,
)
from .view.quiz import (
    summarize_results, aggregate_summary,
    QuizApp, QuestionView,
)


if __name__ == "__main__":  # pragma: no cover
    main()
