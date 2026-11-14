"""Réponses documentaires vérifiées ; les sources et encodeurs sont SYNTHÉTIQUES."""

import json
from types import SimpleNamespace

import pytest

from metropt_maintenance.diagnostic.providers import LLMApiError, MockProvider
from metropt_maintenance.rag import search_documents
from metropt_maintenance.rag.answer import generate_document_answer


def test_document_answer_reuses_mock_and_real_pdf_passages(rag_index):
    docs,index,encoder=rag_index
    retrieval=search_documents('Motor_current',index_directory=index,encoder=encoder)
    answer=generate_document_answer('Motor_current',retrieval,documents_directory=docs,provider=MockProvider())
    assert answer.status=='answered' and answer.simulation and not answer.sufficiency_verified
    assert answer.excerpts[0]['text'] in answer.sources[0].passage
    assert answer.sources[0].page>=1 and answer.sources[0].file_name


def test_no_provider_call_without_documents(rag_index):
    docs,index,encoder=rag_index
    retrieval=search_documents('Quelle recette pour un gateau ?',index_directory=index,encoder=encoder)
    def forbidden(payload): raise AssertionError('Aucun fournisseur sans preuve')
    provider=SimpleNamespace(complete=forbidden)
    answer=generate_document_answer(retrieval.question,retrieval,documents_directory=docs,provider=provider)
    assert answer.status=='information_insuffisante' and not answer.excerpts


@pytest.mark.parametrize('case',['api','invalid','invented_quote','invented_source','unsupported_hypothesis'])
def test_invalid_or_failed_document_answers_are_not_published(rag_index,case):
    docs,index,encoder=rag_index
    retrieval=search_documents('Motor_current',index_directory=index,encoder=encoder)
    def complete(payload):
        if case=='api': raise LLMApiError('Erreur API simulée')
        if case=='invalid': return '{}'
        draft=json.loads(MockProvider().complete(payload))
        if case=='invented_quote': draft['arguments'][0]['quote']='La norme ISO inventee demande une pression de 999 bar.'
        if case=='invented_source': draft['arguments'][0]['source_id']='S8'
        if case=='unsupported_hypothesis': draft['hypotheses']=[{'source_id':'S1','label_quote':'Cause inventee','argument_ids':['A1'],'fact_ids':['F1']}]
        return json.dumps(draft)
    provider=SimpleNamespace(name='test',model='test',simulation=False,complete=complete)
    answer=generate_document_answer(retrieval.question,retrieval,documents_directory=docs,provider=provider)
    assert answer.status==('erreur_api' if case=='api' else 'reponse_invalide')
    assert answer.excerpts==[] and not answer.simulation


def test_modified_pdf_and_different_question_are_rejected(rag_index):
    docs,index,encoder=rag_index
    retrieval=search_documents('Motor_current',index_directory=index,encoder=encoder)
    with pytest.raises(ValueError): generate_document_answer('Autre question',retrieval,documents_directory=docs)
    (docs/retrieval.passages[0].relative_path).unlink()
    answer=generate_document_answer(retrieval.question,retrieval,documents_directory=docs,provider=MockProvider())
    assert answer.status=='sources_invalides'
