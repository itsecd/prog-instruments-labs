#!/usr/bin/env python3

import collections
import dataclasses
import datetime
import git
import json
import os
import subprocess
from typing import Dict, List, Optional, Union

from pydantic import RootModel, TypeAdapter, ValidationError
from typing import Any
from models import frontend, v0, v1

PLANET_INDEXES = range(261)

CACHE_DIR = "_cache"
CACHE_VERSION = 2
RECENCY = 6 * 24

HELDIVERS_FILE = "helldivers.json"
V1_FILE = "801_full_v1.json"
GIT_LOG_FORMAT = "--format=%H"
COMMIT_LIMIT = 1440
CACHE_V0_DIR = "v0"
CACHE_V1_DIR = "v1"
ERROR_KEY = "error"
ERRORS_KEY = "errors"
SKIP_REF_PREFIX = "a514ea"
FORMAT_FILE = ".json"

AGGREGATES_FILE = "./docs/data/aggregates.json"
RECENT_ATTACKS_FILE = "./docs/data/recent_attacks.json"
CURRENT_STATUS_FILE = "./docs/data/current_status.json"


def git_commits_for(path: str) -> List[str]:
    """
    Возвращает список коммитов для указанного пути.

    Args:
        path (str): Путь к файлу или директории в репозитории Git.

    Returns:
        List[str]: Список строк, каждая из которых представляет отдельный коммит.
    """
    return (
        subprocess.check_output(["git", "log", GIT_LOG_FORMAT, path])
        .strip()
        .decode()
        .splitlines()
    )


def git_show(ref: str, name: str, repo_client: Any) -> bytes:
    """
    Получает содержимое файла в указанном коммите из репозитория Git.

    Args:
        ref (str): Ссылка на коммит (например, хэш коммита).
        name (str): Имя файла в дереве коммита.
        repo_client (Any): Клиент для работы с репозиторием, который поддерживает метод commit.

    Returns:
        bytes: Содержимое файла в байтах.
    """
    commit_tree = repo_client.commit(ref).tree

    return commit_tree[name].data_stream.read()


def fetch_all_records_v0() -> List[v0.FullStatus]:
    """
    Извлекает и кэширует записи коммитов, относящиеся к файлу HELDIVERS_FILE, возвращая их в формате v0.FullStatus.

    Извлекает коммиты для HELDIVERS_FILE, затем для каждого коммита:
        - Проверяет наличие данных в кэше и валидирует их.
        - Если данных в кэше нет, загружает из репозитория.
        - Обновляет запись, добавляя timestamp и версию.
        - Кэширует запись для последующего использования.

    Returns:
        List[v0.FullStatus]: Список записей статуса, отсортированных по timestamp.
    """
    commits = git_commits_for(HELDIVERS_FILE)[COMMIT_LIMIT]

    repo = git.Repo(".", odbt=git.db.GitCmdObjectDB)

    out: List[v0.FullStatus] = []

    for ref in commits:
        cache_path = os.path.join(
            CACHE_DIR,
            CACHE_V0_DIR,
            ref[:2],
            ref[2:] + FORMAT_FILE
        )

        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                try:
                    record = TypeAdapter(v0.FullStatus).validate_json(fh.read())
                except ValidationError as exc:
                    print(f"Bad cached data {exc}")
                    continue
                if record.version == CACHE_VERSION:
                    out.append(record)
                    continue
        try:
            record = TypeAdapter(v0.FullStatus).validate_json(
                git_show(ref, HELDIVERS_FILE, repo)
            )
        except ValidationError as exc:
            res = json.loads(git_show(ref, HELDIVERS_FILE, repo))
            if ERROR_KEY in res.keys() or ERRORS_KEY in res.keys():
                continue
            print(f"Bad committed data {exc.errors()[0]}")
        timestamp = repo.commit(ref).committed_datetime.astimezone(
            datetime.timezone.utc
        )
        record.snapshot_at = timestamp
        record.version = CACHE_VERSION

        out.append(record)

        try:
            os.makedirs(os.path.dirname(cache_path))
        except FileExistsError:
            pass
        with open(cache_path, "w") as fh:
            fh.write(RootModel[v0.FullStatus](record).model_dump_json())

    out.sort(key=lambda row: row.snapshot_at)
    return out


def fetch_all_records_v1() -> List[v1.FullStatus]:
    """
    Извлекает и кэширует записи коммитов, относящиеся к файлу V1_FILE, возвращая их в формате v1.FullStatus.

    Извлекает коммиты для V1_FILE, затем для каждого коммита:
        - Проверяет наличие данных в кэше и валидирует их.
        - Если данных в кэше нет, загружает из репозитория.
        - Обновляет запись, добавляя timestamp и версию.
        - Кэширует запись для последующего использования.

    Returns:
        List[v1.FullStatus]: Список записей статуса, отсортированных по timestamp.
    """
    commits = git_commits_for(V1_FILE)[COMMIT_LIMIT]

    repo = git.Repo(".", odbt=git.db.GitCmdObjectDB)

    out: List[v1.FullStatus] = []

    for ref in commits:
        cache_path = os.path.join(
            CACHE_DIR,
            CACHE_V1_DIR,
            ref[:2],
            ref[2:] + FORMAT_FILE
        )

        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                try:
                    record = v1.FullStatus.model_validate_json(fh.read())
                except ValidationError as exc:
                    print(f"Bad cached data {exc}")
                    continue
                if record.version == CACHE_VERSION:
                    out.append(record)
                    continue
        try:
            record = v1.FullStatus.model_validate_json(
                git_show(ref, V1_FILE, repo)
            )
        except ValidationError as exc:
            if ref.startswith(SKIP_REF_PREFIX):
                continue
            res = json.loads(git_show(ref, V1_FILE, repo))
            if ERROR_KEY in res.keys() or ERRORS_KEY in res.keys():
                continue
            print(f"Bad committed data {exc.errors()[0]}")
        timestamp = repo.commit(ref).committed_datetime.astimezone(
            datetime.timezone.utc
        )
        record.snapshot_at = timestamp
        record.version = CACHE_VERSION

        out.append(record)

        try:
            os.makedirs(os.path.dirname(cache_path))
        except FileExistsError:
            pass
        with open(cache_path, "w") as fh:
            fh.write(record.model_dump_json())

    out.sort(key=lambda row: row.snapshot_at)
    return out


def create_agg_stats() -> None:
    """
    Создает агрегированные статистические данные на основе записей, полученных из fetch_all_records_v1().

    Обрабатывает статистику для всех записей, в том числе:
        - количество игроков,
        - события на планетах,
        - активные планеты и их истории,
        - и наиболее активные планеты за последнее время.

    Сохраняет результаты в файлы:
        - AGGREGATES_FILE: для основной статистики (включая временные метки, игроков и влияние).
        - RECENT_ATTACKS_FILE: для записи наиболее активных планет.
        - CURRENT_STATUS_FILE: для сохранения последнего состояния.

    Returns:
        None
    """
    records = [v1_to_frontend(rec) for rec in fetch_all_records_v1()]
    players = [0] * len(records)
    timestamps = []
    time_correction = []
    impact = []
    active = set(
        [campaign.planet.index
        for campaign in records[len(records) - 1].active]
    )
    active_sum = {p: 0 for p in active}
    active_planet_hist = []

    recent_start = len(records) - (RECENCY)
    for (step, record) in enumerate(records):
        active_step = {}
        for status in record.planets:
            players[step] += status.statistics.player_count
            if status.index in active:
                active_step[status.index] = {
                    "players": status.statistics.player_count,
                    "liberation": status.liberation,
                }
                if step > recent_start:
                    active_sum[status.index] += status.statistics.player_count
        for event in record.events:
            planet = record.planets[event.planet.index]
            active_step[event.planet.index] = {
                "players": planet.statistics.player_count,
                "liberation": event.liberation,
            }
        active_planet_hist.append(active_step)

    most_active = sorted(active_sum.items(), key=lambda x: x[1], reverse=True)

    for step in records:
        timestamps.append(step.snapshot_at)
        impact.append(step.war.impact_multiplier)

    with open(AGGREGATES_FILE, "w") as fh:
        json.dump(
            [
                {"timestamp": v1, "players": v2, "impact": v3, "attacks": v4}
                for v1, v2, v3, v4 in zip(
                    timestamps, players, impact, active_planet_hist
                )
            ],
            fh,
        )
    with open(RECENT_ATTACKS_FILE, "w") as fh:
        json.dump(most_active, fh)
    with open(CURRENT_STATUS_FILE, "w") as fh:
        fh.write(records[-1].model_dump_json())


def wrap_if_str(val: Any) -> Union[Dict[str, str], Any]:
    """
    Оборачивает строку в словарь с ключом 'en-US'. 

    Если входное значение — строка, возвращает словарь с ключом 'en-US' и значением этой строки.
    Если входное значение не является строкой, возвращает его без изменений.

    Args:
        val (Any): Значение для проверки.

    Returns:
        Union[Dict[str, str], Any]: Словарь с ключом 'en-US' и строковым значением, 
                                    либо исходное значение, если оно не является строкой.
    """
    if isinstance(val, str):
        return {"en-US": val}
    return val


def v1_to_frontend(v1_rec: v1.FullStatus) -> frontend.CurrentStatus:
    """
    Преобразует данные статуса v1.FullStatus в формат frontend.CurrentStatus для интерфейса.

    Извлекает и преобразует данные из v1.FullStatus:
        - Масштабирует координаты планет.
        - Конвертирует временные метки в миллисекунды.
        - Оборачивает строки, требующие многоязычной поддержки, в словари с ключом 'en-US'.
        - Создает необходимые объекты для планет, событий, кампаний, заданий и сообщений.

    Args:
        v1_rec (v1.FullStatus): Объект статуса в формате v1.FullStatus для преобразования.

    Returns:
        frontend.CurrentStatus: Объект статуса для отображения на фронтенде.
    """
    planets = []
    events = []

    for planet in v1_rec.planets:
        planet.name = wrap_if_str(planet.name)
        planet.position.x *= 100
        planet.position.y *= 100
        planets.append(frontend.Planet.model_validate(planet.model_dump()))
        if planet.event is not None:
            events.append(
                frontend.Defense.model_validate(
                    {
                        "id": planet.event.id,
                        "faction": planet.event.faction,
                        "type": planet.event.event_type,
                        "start_time": int(planet.event.start_time.timestamp() * 1000),
                        "end_time": int(planet.event.end_time.timestamp() * 1000),
                        "health": planet.event.health,
                        "max_health": planet.event.max_health,
                        "joint_operation_ids": planet.event.joint_operation_ids,
                        "planet": planet.model_dump(),
                    }
                )
            )

    assignments = []
    for assignment in v1_rec.assignments:
        assignments.append(
            frontend.Assignment.model_validate(
                {
                    "id": assignment.id,
                    "title": wrap_if_str(assignment.title),
                    "briefing": wrap_if_str(assignment.briefing),
                    "description": wrap_if_str(assignment.description),
                    "tasks": [task.model_dump() for task in assignment.tasks],
                    "reward": assignment.reward.model_dump(),
                    "progress": assignment.progress,
                    "expiration": int(assignment.expiration.timestamp() * 1000),
                }
            )
        )
    stats = v1_rec.war.statistics
    war_details = frontend.WarDetails.model_validate(
        {
            "start_time": int(v1_rec.war.started.timestamp() * 1000),
            "end_time": int(v1_rec.war.ended.timestamp() * 1000),
            "now": int(v1_rec.war.now.timestamp() * 1000),
            "factions": v1_rec.war.factions,
            "impact_multiplier": v1_rec.war.impact_multiplier,
            "statistics": frontend.Statistics.model_validate(
                v1_rec.war.statistics.model_dump()
            ),
        }
    )

    campaigns = []
    for campaign in v1_rec.campaigns:
        campaign.planet.name = wrap_if_str(campaign.planet.name)
        campaign.planet.position.x *= 100
        campaign.planet.position.y *= 100
        campaigns.append(frontend.Campaign.model_validate(campaign.model_dump()))

    dispatches: List[frontend.Dispatch] = []
    for dispatch in v1_rec.dispatches:
        if dispatch.message is None:
            continue
        dispatches.append(
            frontend.Dispatch.model_validate(
                {
                    "id": dispatch.id,
                    "message": wrap_if_str(dispatch.message),
                    "title": wrap_if_str("Dispatch"),
                }
            )
        )

    return frontend.CurrentStatus.model_validate(
        {
            "events": events,
            "planets": planets,
            "assignments": assignments,
            "war": war_details,
            "active": campaigns,
            "dispatches": dispatches,
            "snapshot_at": int(v1_rec.snapshot_at.timestamp() * 1000),
        }
    )


def v0_to_frontend(v0_rec: v0.FullStatus) -> frontend.CurrentStatus:
    """
    Преобразует данные статуса v0.FullStatus в формат frontend.CurrentStatus для интерфейса.

    Извлекает и преобразует данные из v0.FullStatus:
        - Конвертирует временные метки в миллисекунды.
        - Оборачивает строки с именами планет в словари с ключом 'en-US'.
        - Создает объекты для планет, событий, кампаний и войн с соответствующими полями.
        - Считает общее количество игроков.

    Args:
        v0_rec (v0.FullStatus): Объект статуса в формате v0.FullStatus для преобразования.

    Returns:
        frontend.CurrentStatus: Объект статуса для отображения на фронтенде.
    """
    events = []
    planets: List[frontend.Planet] = []
    total_players = 0
    for planet_status in v0_rec.planet_status:
        planets.append(
            frontend.Planet.model_validate(
                {
                    "position": planet_status.planet.position,
                    "index": planet_status.planet.index,
                    "name": {"en-US": planet_status.planet.name},
                    "sector": planet_status.planet.sector,
                    "waypoints": planet_status.planet.waypoints,
                    "disabled": planet_status.planet.disabled,
                    "regen_per_second": planet_status.regen_per_second,
                    "current_owner": planet_status.owner,
                    "initial_owner": planet_status.planet.initial_owner,
                    "health": planet_status.health,
                    "max_health": planet_status.planet.max_health,
                    "statistics": frontend.Statistics.model_validate(
                        {"player_count": planet_status.players}
                    ),
                    "attacking": [
                        target
                        for (source, target) in v0_rec.planet_attacks
                        if source == planet_status.planet.index
                    ],
                }
            )
        )
        total_players += planet_status.players

    for event in v0_rec.planet_events:
        if event.event_type == 1:
            events.append(
                frontend.Defense.model_validate(
                    {
                        "id": event.id,
                        "faction": event.race,
                        "type": event.event_type,
                        "start_time": int(event.start_time.timestamp() * 1000),
                        "end_time": int(event.expire_time.timestamp() * 1000),
                        "health": event.health,
                        "max_health": event.max_health,
                        "joint_operation_ids": [
                            j["id"] for j in event.joint_operations
                        ],
                        "planet": next(
                            filter(
                                lambda p: p.index == event.planet.index,
                                planets
                                )
                        ),
                    }
                )
            )

    war_details = frontend.WarDetails.model_validate(
        {
            "start_time": int(v0_rec.started_at.timestamp() * 1000),
            "end_time": None,
            "now": int(v0_rec.snapshot_at.timestamp() * 1000),
            "factions": ["Humans", "Automatons", "Terminids", "Illuminate"],
            "impact_multiplier": v0_rec.impact_multiplier,
            "statistics": frontend.Statistics.model_validate(
                {"player_count": total_players}
            ),
        }
    )

    campaigns = []
    for campaign in v0_rec.campaigns:
        campaigns.append(
            frontend.Campaign.model_validate(
                {
                    "count": campaign.count,
                    "id": campaign.id,
                    "planet": next(
                        filter(
                            lambda p: p.index == campaign.planet.index, 
                            planets
                        )
                    ),
                    "type": campaign.type,
                }
            )
        )

    return frontend.CurrentStatus.model_validate(
        {
            "events": events,
            "planets": planets,
            "assignments": [],
            "war": war_details,
            "active": campaigns,
            "dispatches": [dataclasses.asdict(d)
            for d in v0_rec.global_events],
            "snapshot_at": war_details.now,
        }
    )


if __name__ == "__main__":
    create_agg_stats()
# Plotting recent attacks based solely on player count is a bit boring sometimes.
# Maybe we should use variance of liberation?

# intial_owner's do change, such as when we lost the defense of Angel's Venture.
# We should keep track of these and add them to the message logs