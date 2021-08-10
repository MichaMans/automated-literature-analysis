# -*- coding: utf-8 -*-
# @Author: MichaMans
# @Date:   2021-07-20 14:40:37
# @Last Modified by:   MichaMans
# @Last Modified time: 2021-08-10 09:49:44

import os

# from scholarly import scholarly, ProxyGenerator
import metaknowledge as mk
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import litstudy
import streamlit as st


def main():

    st.set_option('deprecation.showPyplotGlobalUse', False)

    plt.style.use('~/ebc.paper.mplstyle')
    sns.set()
    sns.set_context("paper")
    colors = ["#00549F", "#407FB7", "#8EBAE5", "#C7DDF2", "#E8F1FA"]
    sns.set_palette(sns.color_palette(colors))

    dir_this = os.path.abspath(os.path.dirname(__file__))
    code_dir = os.path.abspath(os.path.dirname(dir_this))
    data_dir = os.path.join(code_dir, "data")

    # scopus_file_bibtex = os.path.join(
    #     data_dir, "search_term_2_scopus_315entries.bib")

    query_file = os.path.join(
        data_dir, "search_terms_scopus_ready_to_read.txt")

    bigrams_file = os.path.join(data_dir, "bigrams.txt")
    stopwords_file = os.path.join(data_dir, "stopwords.txt")

    # load from scorpus via search term
    search_queries = load_queries_from_file(query_file)
    scopus_query = search_queries["search_1"]
    # scopus_query = 'TITLE-ABS-KEY("low temperature network*" OR "ultra low temperature network*" OR "low temperature grid*" OR "ultra low temperature grid*" OR "5th generation grid*" OR "5th generation network*" OR "5th generation district heating and cooling" OR "fifth generation grid*" OR "fifth generation network*" OR "fifth generation district heating and cooling" OR "fith-generation grid*" OR "fith-generation network*" OR "fith-generation district heating and cooling") AND TITLE-ABS-KEY("district*" OR "energ*" OR "heat*" OR "cool*")'
    docset = search_scopus(scopus_query)

    st.text("Now analyzing %s items!"%len(docset))

    # load from bibtex
    # docset = load_bibtex(scopus_file_bibtex, lookup_authors=False)

    # working
    fig1, ax1 = litstudy.prepare_plot(15, 3)
    fig1 = litstudy.plot_year_histogram(docset, ax=ax1)
    st.pyplot(fig1)

    # working
    fig2, ax2 = litstudy.prepare_plot(10, 10)
    fig2 = litstudy.plot_source_histogram(docset, ax=ax2, top_k=50, clean=False)
    st.pyplot(fig2)

    # working
    fig3, ax3 = litstudy.prepare_plot(10, 3)
    fig3 = litstudy.plot_source_type_histogram(docset, ax=ax3)
    st.pyplot(fig3)

    # WORKING
    fig4, ax4 = litstudy.prepare_plot(10, 3)
    fig4 = litstudy.plot_language_histogram(docset)
    st.pyplot(fig4)

    # working
    fig5, ax5 = litstudy.prepare_plot(10, 3)
    fig5 = litstudy.plot_number_authors_histogram(docset)
    st.pyplot(fig5)

    # working
    fig6, ax6 = litstudy.prepare_plot(10, 15)
    fig6 = litstudy.plot_author_histogram(docset, top_k=75)
    st.pyplot(fig6)

    # WORKING
    fig7, ax7 = litstudy.prepare_plot(10, 15)
    fig7 = litstudy.plot_author_affiliation_histogram(docset, top_k=75)
    st.pyplot(fig7)

    # WORKING
    fig8, ax8 = litstudy.prepare_plot(10, 15)
    fig8 = litstudy.plot_affiliation_histogram(docset, top_k=75, clean=False)
    st.pyplot(fig8)

    # WORKING
    fig9, ax9 = litstudy.prepare_plot(10, 15)
    fig9 = litstudy.plot_country_histogram(docset)
    st.pyplot(fig9)

    # WORKING
    fig10, ax10 = litstudy.prepare_plot(10, 15)
    fig10 = litstudy.plot_affiliation_type_histogram(docset)
    st.pyplot(fig10)

    # WORKING
    fig11, ax11 = litstudy.prepare_plot(20, 20)
    fig11 = litstudy.plot_citation_network(docset)
    st.pyplot(fig11)

    # WORKING
    # fig12, ax12 = litstudy.prepare_plot(5, 5)
    fig12 = litstudy.plot_coauthor_network(docset, min_degree=3)
    st.pyplot(fig12)

    # working
    # Filter documents that have either no abstract or a short abstract (less than 50 characters)
    filtered_docset = docset.filter(
        lambda d: d.abstract is not None and len(d.abstract) >= 50)

    dic, freqs = litstudy.nlp.build_corpus_simple(
        filtered_docset, bigrams=bigrams_file, stopwords=stopwords_file)

    fig13, ax13 = litstudy.prepare_plot(15, 20)
    fig13 = litstudy.plot_words_histogram(freqs, dic, top_k=100)
    st.pyplot(fig13)

    nmf_model = litstudy.nlp.train_nmf_model(dic, freqs, num_topics=9)

    fig14, ax14 = litstudy.prepare_plot(10, 10)
    fig14 = litstudy.plot_topic_clouds(nmf_model, cols=3)
    st.pyplot(fig14)

    fig15, ax15 = litstudy.prepare_plot(15, 15)
    fig15 = litstudy.plot_topic_map(nmf_model, dic, freqs)
    st.pyplot(fig15)

    for i in range(len(filtered_docset)):
        if nmf_model.doc2topic[i, 3] > 0.5:
            print(filtered_docset[i].title)


@st.cache
def load_bibtex(file, lookup_authors):
    """plots and saves number of modelica publications"""
    docset = litstudy.load_bibtex(file, lookup_authors=lookup_authors)

    return docset


@st.cache
def search_scopus(scopus_query):
    """plots and saves number of modelica publications"""
    docset = litstudy.search_scopus(scopus_query)

    return docset


def load_queries_from_file(filepath):
    """plots and saves number of modelica publications"""
    with open(filepath, 'r') as file:
        data = file.read()

        query_list = data.replace("\t", "").replace("\n", "").split("###")
        # remove breakpoint strings
        query_list = [i for i in query_list if i]
        # data = file.read().replace('\n', '')
        query_tuples = [(
            query_list[i], query_list[i + 1]) for i in range(0, len(query_list), 2)]
        queries = dict(query_tuples)
    return queries


def make_workspace(name_workspace=None):
    """Creates a local workspace with given name

    If no name is given, the general workspace directory will be used

    Parameters
    ----------
    name_workspace : str
        Name of the local workspace to be created

    Returns
    -------
    workspace : str
        Full path to the new workspace
    """
    dir_this = os.path.abspath(os.path.dirname(__file__))
    code_dir = os.path.abspath(os.path.dirname(dir_this))
    workspace = os.path.join(code_dir, "workspace", "literature_overview")
    if not os.path.exists(workspace):
        os.mkdir(workspace)

    if name_workspace is not None:
        workspace = os.path.join(workspace, name_workspace)
        if not os.path.exists(workspace):
            os.mkdir(workspace)

    return workspace


# Main function
if __name__ == "__main__":
    print("*** Generating literature analysis ***")
    main()
    print("*** Done ***")
