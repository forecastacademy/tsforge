.PHONY: install lint format test docs serve nbclean

install:
\tpoetry install

lint:
\tpoetry run ruff check .

format:
\tpoetry run ruff format .

test:
\tpoetry run pytest -q

docs:
\tpoetry run mkdocs build

serve:
\tpoetry run mkdocs serve

nbclean:
\tpoetry run nbstripout --install
