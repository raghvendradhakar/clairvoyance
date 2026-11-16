-- Migration: Add GUARDRAIL to the evaluation_type enum
-- Description: per-agent live chat guardrail (input + output checks). No rows
-- seeded; the configuration POST creates the row, disabled.
-- The CHECK stays shape-based (as in 082) and adds a third shape:
-- 'execution' string + 'input' and 'output' objects.

ALTER TYPE evaluation_type ADD VALUE IF NOT EXISTS 'GUARDRAIL';

ALTER TABLE evaluation_config
    DROP CONSTRAINT evaluation_config_runtime_check;

ALTER TABLE evaluation_config
    ADD CONSTRAINT evaluation_config_runtime_check
        CHECK (
            (
                jsonb_typeof(configuration -> 'model') = 'string'
                AND btrim(COALESCE(configuration ->> 'model', '')) <> ''
                AND jsonb_typeof(configuration -> 'system_prompt') = 'string'
                AND btrim(COALESCE(configuration ->> 'system_prompt', '')) <> ''
            )
            OR (
                jsonb_typeof(configuration -> 'engine') = 'string'
                AND btrim(COALESCE(configuration ->> 'engine', '')) <> ''
                AND jsonb_typeof(configuration -> 'provider') = 'string'
                AND btrim(COALESCE(configuration ->> 'provider', '')) <> ''
                AND jsonb_typeof(configuration -> 'model') = 'string'
                AND btrim(COALESCE(configuration ->> 'model', '')) <> ''
                AND jsonb_typeof(configuration -> 'questions') = 'array'
            )
            OR (
                jsonb_typeof(configuration -> 'execution') = 'string'
                AND jsonb_typeof(configuration -> 'input') = 'object'
                AND jsonb_typeof(configuration -> 'output') = 'object'
            )
        );
