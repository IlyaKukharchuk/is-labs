### Как запустить код:
1. Убедитесь, что структура папок соответствует:
   ```
   data/
       Толстой/
           text1.txt
           text2.txt
           ...
       Достоевский/
           text1.txt
           ...
       Чехов/
           text1.txt
           ...
   ```

2. Установите зависимости:
   ```bash
   pip install spacy pandas scikit-learn gensim tensorflow matplotlib wordcloud
   python -m spacy download ru_core_news_sm
   ```

3. Поместите входной текст для анализа в файл `Булгаков2.txt`.

4. Запустите скрипт:
   ```bash
   python main7.py
   ```

---

### Если возникают ошибки:
1. **UnicodeDecodeError**: Убедитесь, что все текстовые файлы в папке `data` имеют кодировку `utf-8` или `cp1251`.
2. **MemoryError**: Уменьшите `max_length` в функции `preprocess` или используйте более легкие модели.
3. **Отсутствие файлов**: Проверьте наличие файла `Булгаков2.txt` или измените имя файла в коде.