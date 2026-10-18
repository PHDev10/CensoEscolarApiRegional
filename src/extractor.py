import logging
import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = (
    BASE_DIR / "datasets" / "dados_censo_escolar" / "Tabela_Matricula_2025_V2.csv"
)
DB_PATH = BASE_DIR / "datasets" / "censo_escolar.db"

CHUNKSIZE = 50_000
SEPARATOR = ";"
ENCODING_CANDIDATES = ("utf-8", "latin1")
TABLE = "instituicoes"

COLUMN_MAP = {
    "NO_ENTIDADE": "no_entidade",
    "CO_ENTIDADE": "co_entidade",
    "NO_UF": "no_uf",
    "SG_UF": "sg_uf",
    "CO_UF": "co_uf",
    "NO_MUNICIPIO": "no_municipio",
    "CO_MUNICIPIO": "co_municipio",
    "NU_ANO_CENSO": "nu_ano_censo",
    "QT_MAT_BAS": "qt_mat_bas",
    "QT_MAT_INF": "qt_mat_inf",
    "QT_MAT_FUND": "qt_mat_fund",
    "QT_MAT_MED": "qt_mat_med",
    "QT_MAT_PROF": "qt_mat_prof",
    "QT_MAT_EJA": "qt_mat_eja",
    "QT_MAT_ESP": "qt_mat_esp",
}

SIGLAS_NORDESTE = ("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE")
SIGLAS_SUDESTE = ("ES", "MG", "RJ", "SP")
CO_UF_NORDESTE = range(21, 30)  # 21..29
CO_UF_SUDESTE = range(31, 36)  # 31..35

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etl-censo")


def validate_source_columns(csv_path: Path, sep: str, encoding: str) -> None:
    """Valida a presença de todas as colunas obrigatórias no CSV de origem."""
    header = pd.read_csv(csv_path, sep=sep, encoding=encoding, nrows=0).columns.tolist()
    missing = [c for c in COLUMN_MAP if c not in header]
    if missing:
        raise ValueError(f"Colunas ausentes no arquivo {csv_path.name}: {missing}")
    log.info(
        "Validação OK: todas as %d colunas obrigatórias presentes.", len(COLUMN_MAP)
    )


def detect_encoding(csv_path: Path) -> str:
    for encoding in ENCODING_CANDIDATES:
        try:
            pd.read_csv(csv_path, sep=SEPARATOR, encoding=encoding, nrows=5)
            log.info("Encoding detectado: %s", encoding)
            return encoding
        except UnicodeDecodeError:
            log.info("Tentativa com %s falhou; testando próximo encoding...", encoding)
    raise UnicodeDecodeError(
        b"extractor",
        b"",
        0,
        0,
        "Não foi possível decodificar o CSV com os encodings de teste.",
    )


def filter_regioes(df: pd.DataFrame) -> pd.DataFrame:
    """O filtro é feito por SG_UF (sigla) com fallback por CO_UF (código IBGE):
    Nordeste: AL, BA, CE, MA, PB, PE, PI, RN, SE (CO_UF 21-29)
    Sudeste:  ES, MG, RJ, SP (CO_UF 31-35)
    """
    sg_uf = df["SG_UF"].astype(str).str.strip().str.upper()
    co_uf = pd.to_numeric(df["CO_UF"], errors="coerce")

    nordeste = sg_uf.isin(SIGLAS_NORDESTE) | co_uf.isin(CO_UF_NORDESTE)
    sudeste = sg_uf.isin(SIGLAS_SUDESTE) | co_uf.isin(CO_UF_SUDESTE)
    return df[nordeste | sudeste]


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas, adiciona flag `ativo` e ajusta tipos para carga."""
    df = df.rename(columns=COLUMN_MAP)
    df["ativo"] = 1  # exclusão lógica

    numeric_cols = [
        "co_entidade",
        "co_uf",
        "co_municipio",
        "nu_ano_censo",
        "qt_mat_bas",
        "qt_mat_inf",
        "qt_mat_fund",
        "qt_mat_med",
        "qt_mat_prof",
        "qt_mat_eja",
        "qt_mat_esp",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    text_cols = ["no_entidade", "no_uf", "sg_uf", "no_municipio"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    return df[list(COLUMN_MAP.values())]


def create_table_if_needed(conn: sqlite3.Connection) -> None:
    """Cria a tabela `instituicoes` com a flag `ativo` (padrão 1)."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            no_entidade TEXT NOT NULL,
            co_entidade INTEGER NOT NULL,
            no_uf TEXT,
            sg_uf TEXT,
            co_uf INTEGER,
            no_municipio TEXT,
            co_municipio INTEGER,
            nu_ano_censo INTEGER,
            qt_mat_bas INTEGER,
            qt_mat_inf INTEGER,
            qt_mat_fund INTEGER,
            qt_mat_med INTEGER,
            qt_mat_prof INTEGER,
            qt_mat_eja INTEGER,
            qt_mat_esp INTEGER,
            ativo INTEGER NOT NULL DEFAULT 1
        )
        """)
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_co_entidade ON {TABLE}(co_entidade)"
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_sg_uf ON {TABLE}(sg_uf)")
    conn.commit()


def load_dataframe(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Insere o bloco tratado no SQLite"""
    df = df.astype({c: "object" for c in df.select_dtypes("Int64").columns})
    df = df.where(pd.notna(df), None)
    conn.executemany(
        f"INSERT INTO {TABLE} ({', '.join(df.columns)}) VALUES "
        f"({', '.join('?' * len(df.columns))})",
        df.itertuples(index=False, name=None),
    )
    conn.commit()


def main() -> int:
    if not CSV_PATH.is_file():
        log.error(
            "Arquivo de origem não encontrado: %s\n"
            "Baixe os microdados do Censo Escolar 2025 no INEP e salve-o em "
            "datasets/dados_censo_escolar/ para rodar o ETL.",
            CSV_PATH,
        )
        return 1

    encoding = detect_encoding(CSV_PATH)

    validate_source_columns(CSV_PATH, SEPARATOR, encoding)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    total_lines_read = 0
    total_inserted = 0
    totals_by_uf: dict[str, int] = {}

    try:
        log.info("Iniciando extração paginada de %s", CSV_PATH.name)
        create_table_if_needed(conn)

        chunks = pd.read_csv(
            CSV_PATH,
            sep=SEPARATOR,
            encoding=encoding,
            chunksize=CHUNKSIZE,
            low_memory=False,
            usecols=list(COLUMN_MAP.keys()),
        )

        for i, chunk in enumerate(chunks, start=1):
            try:
                filtered = filter_regioes(chunk)
                prepared = prepare_dataframe(filtered)
                total_lines_read += len(chunk)
                total_inserted += len(prepared)

                for uf, count in prepared["sg_uf"].value_counts().items():
                    totals_by_uf[uf] = totals_by_uf.get(uf, 0) + int(count)

                if not prepared.empty:
                    load_dataframe(conn, prepared)

                log.info(
                    "Bloco %d: linhas lidas no bloco=%d, filtradas/inseridas=%d, "
                    "total inserido até agora=%d",
                    i,
                    len(chunk),
                    len(prepared),
                    total_inserted,
                )
            except Exception:
                log.exception("Falha ao processar o bloco %d.", i)
                conn.rollback()
                raise

        log.info(
            "Concluído: %d linhas lidas, %d instituições filtradas e inseridas.",
            total_lines_read,
            total_inserted,
        )
        log.info("Por UF: %s", dict(sorted(totals_by_uf.items())))
        log.info("Banco gerado em: %s", DB_PATH)
        return 0

    except Exception:
        log.exception("Erro fatal no pipeline de extração.")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
