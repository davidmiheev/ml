"""Synthetic episodic corpus in the spirit of the contest task (dev-only proxy).

Every message carries latent attributes; an episode hides one binary rule over
those attributes and contrasts four positives with four hard negatives that
share the remaining attributes.  Rules used for validation/test are disjoint
from the training rules so the proxy measures generalisation to unseen rules.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TOPICS = {
    "delivery": ["доставка", "курьер", "посылка", "заказ"],
    "payment": ["оплата", "платёж", "транзакция", "счёт"],
    "account": ["аккаунт", "профиль", "пароль", "вход"],
    "support": ["поддержка", "оператор", "заявка", "тикет"],
    "refund": ["возврат", "компенсация", "отмена", "чек"],
    "subscription": ["подписка", "тариф", "продление", "списание"],
    "food": ["ресторан", "меню", "блюдо", "кухня"],
    "travel": ["билет", "рейс", "отель", "поездка"],
}

TOPIC_OBJECT = {
    "delivery": ["посылку", "заказ", "курьера", "доставку"],
    "payment": ["платёж", "оплату", "списание", "счёт"],
    "account": ["аккаунт", "профиль", "пароль", "вход в личный кабинет"],
    "support": ["заявку", "обращение", "тикет", "ответ оператора"],
    "refund": ["возврат", "компенсацию", "отмену заказа", "чек"],
    "subscription": ["подписку", "тариф", "продление", "автосписание"],
    "food": ["заказ еды", "меню", "блюдо", "бронь столика"],
    "travel": ["билет", "рейс", "бронь отеля", "поездку"],
}

QUESTION_TEMPLATES = [
    "Подскажите, когда придёт {obj}?",
    "Можно уточнить статус про {obj}?",
    "Кто отвечает за {obj}?",
    "Как быстро оформить {obj}?",
    "Что делать, если {obj} задерживается?",
]
COMPLAINT_TEMPLATES = [
    "Опять сорвали {obj}, это никуда не годится.",
    "Уже третий раз проблема через {obj}.",
    "Крайне недоволен тем, как обработали {obj}.",
    "Из-за вас сорвался {obj}, верните деньги.",
    "Совершенно ужасно оформили {obj}.",
]
THANKS_TEMPLATES = [
    "Спасибо, {obj} оформили очень быстро.",
    "Благодарю за помощь, {obj} наконец на месте.",
    "Отличная работа, {obj} пришёл раньше срока.",
    "Всё супер, {obj} полностью устроил.",
    "Приятно удивлён, {obj} сделали идеально.",
]
REQUEST_TEMPLATES = [
    "Пожалуйста, проверьте {obj} ещё раз.",
    "Прошу ускорить {obj} по возможности.",
    "Нужно переоформить {obj} на другое имя.",
    "Оформите, пожалуйста, {obj} заново.",
    "Давайте перенесём {obj} на завтра.",
]
NOTICE_TEMPLATES = [
    "Сообщаю, что {obj} уже получен.",
    "Информирую: {obj} обработан без замечаний.",
    "Для протокола: {obj} закрыт сегодня утром.",
    "Уведомляю, что {obj} перешёл в архив.",
    "Фиксирую: {obj} подтверждён системой.",
]
ACT_TEMPLATES = {
    "question": QUESTION_TEMPLATES,
    "complaint": COMPLAINT_TEMPLATES,
    "thanks": THANKS_TEMPLATES,
    "request": REQUEST_TEMPLATES,
    "notice": NOTICE_TEMPLATES,
}
ACT_SENTIMENT = {
    "question": "neutral",
    "complaint": "negative",
    "thanks": "positive",
    "request": "neutral",
    "notice": "neutral",
}
FORMAL_PREFIX = ["Уважаемая служба поддержки, ", "Добрый день, ", "Здравствуйте, "]
INFORMAL_PREFIX = ["Слушайте, ", "Народ, ", "Эй, ", "Короче, "]
URGENT_SUFFIX = [" Это срочно!", " Нужно сегодня же.", " Горит дедлайн.", " Очень срочно."]
NUMBER_SUFFIX = [" Номер {n}.", " Заказ #{n}.", " Сумма {n} рублей.", " Код {n}."]


def make_message(rng: random.Random, attrs: dict) -> str:
    obj = rng.choice(TOPIC_OBJECT[attrs["topic"]])
    text = rng.choice(ACT_TEMPLATES[attrs["act"]]).format(obj=obj)
    if attrs["formal"]:
        text = rng.choice(FORMAL_PREFIX) + text[0].lower() + text[1:]
    elif attrs["informal"]:
        text = rng.choice(INFORMAL_PREFIX) + text[0].lower() + text[1:]
    if attrs["urgent"]:
        text = text + rng.choice(URGENT_SUFFIX)
    if attrs["number"]:
        text = text + rng.choice(NUMBER_SUFFIX).format(n=rng.randint(100, 99999))
    return " ".join(text.split())


def sample_attrs(rng: random.Random, **fixed) -> dict:
    attrs = {
        "topic": rng.choice(list(TOPICS)),
        "act": rng.choice(list(ACT_TEMPLATES)),
        "formal": rng.random() < 0.35,
        "informal": False,
        "urgent": rng.random() < 0.3,
        "number": rng.random() < 0.3,
    }
    attrs.update(fixed)
    if attrs["formal"]:
        attrs["informal"] = False
    attrs["sentiment"] = ACT_SENTIMENT[attrs["act"]]
    return attrs


# A rule is (name, predicate, sampler_for_positive, sampler_for_negative)
def build_rules() -> list[tuple[str, str, object]]:
    rules = []
    for topic in TOPICS:
        rules.append(("topic:" + topic, "topic", topic))
    for act in ACT_TEMPLATES:
        rules.append(("act:" + act, "act", act))
    for sentiment in ("positive", "negative", "neutral"):
        rules.append(("sent:" + sentiment, "sentiment", sentiment))
    rules.append(("urgent", "urgent", True))
    rules.append(("number", "number", True))
    rules.append(("formal", "formal", True))
    return rules


def sample_for_rule(rng: random.Random, field: str, value, positive: bool, shared: dict) -> dict:
    for _ in range(200):
        attrs = sample_attrs(rng, **shared)
        if field == "sentiment":
            ok = attrs["sentiment"] == value
        else:
            ok = attrs[field] == value
        if ok == positive:
            return attrs
    return attrs


def make_episode(rng: random.Random, rule, episode_index: int, labeled: bool,
                 query_balance: str = "fixed") -> dict:
    _, field, value = rule
    shared = {}
    # hard negatives: freeze one confounding attribute across the whole episode
    if field != "topic" and rng.random() < 0.7:
        shared["topic"] = rng.choice(list(TOPICS))
    if field not in ("act", "sentiment") and rng.random() < 0.3:
        shared["act"] = rng.choice(list(ACT_TEMPLATES))
    if query_balance == "fixed":
        n_pos_q = 2
    else:
        n_pos_q = rng.choices([0, 1, 2, 3, 4], weights=[1, 4, 6, 4, 1])[0]
    texts_pos, texts_neg = set(), set()
    while len(texts_pos) < 4 + n_pos_q:
        texts_pos.add(make_message(rng, sample_for_rule(rng, field, value, True, shared)))
    while len(texts_neg) < 8 - n_pos_q:
        texts_neg.add(make_message(rng, sample_for_rule(rng, field, value, False, shared)))
    pos = rng.sample(sorted(texts_pos), 4 + n_pos_q)
    neg = rng.sample(sorted(texts_neg), 8 - n_pos_q)
    support = [{"example_id": f"s{i}", "text": t, "label": 1} for i, t in enumerate(pos[:4])]
    support += [{"example_id": f"s{i+4}", "text": t, "label": 0} for i, t in enumerate(neg[:4])]
    rng.shuffle(support)
    for i, item in enumerate(support):
        item["example_id"] = f"s_{episode_index}_{i}"
    queries = [{"query_id": f"q{i}", "text": t, "label": 1} for i, t in enumerate(pos[4:])]
    queries += [{"query_id": f"q{i+n_pos_q}", "text": t, "label": 0} for i, t in enumerate(neg[4:])]
    rng.shuffle(queries)
    for i, item in enumerate(queries):
        item["query_id"] = f"q_{episode_index}_{i}"
        if not labeled:
            item.pop("label")
    return {"episode_id": f"ep_{episode_index}", "support": support, "queries": queries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("dev/data"))
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--valid", type=int, default=600)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--query-balance", choices=("fixed", "free"), default="fixed")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rules = build_rules()
    rng.shuffle(rules)
    holdout = rules[: max(3, len(rules) // 4)]
    seen = rules[len(holdout) :]
    args.out.mkdir(parents=True, exist_ok=True)
    index = 0
    for name, count, pool, labeled in (
        ("train", args.train, seen, True),
        ("valid", args.valid, holdout, True),
        ("test", args.test, holdout, True),
    ):
        path = args.out / f"{name}.jsonl"
        rule_names = []
        with path.open("w", encoding="utf-8") as stream:
            for _ in range(count):
                rule = rng.choice(pool)
                episode = make_episode(rng, rule, index, labeled, args.query_balance)
                rule_names.append(rule[0])
                index += 1
                stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
        (args.out / f"{name}.rules.json").write_text(json.dumps(rule_names), encoding="utf-8")
        print(name, count, path)
    print("held-out rules:", [r[0] for r in holdout])


if __name__ == "__main__":
    main()
